import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import time
torch.manual_seed(3407)
import torch.nn as nn
import wandb
from etseed.utils.loss_utils import compute_loss
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
from pymunk.space_debug_draw_options import SpaceDebugColor
from skvideo.io import vwrite
from pdb import set_trace as bp
from etseed.dataset.toy_dataset import ToyDataset
from etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant, SE3ManiNet_Equivariant_Separate
from etseed.utils.SE3diffusion_scheduler import DiffusionScheduler, DiffusionScheduler_vanilla
from etseed.utils.group_utils import SE3_to_se2,se2_to_SE3

# Set the environment variables.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(3407)

# Configure parameters
config = {
    "dataset_path": "/home/yue.chen/work/Robotics/SE3-EquivManip/log/50_rotate_triangle.npy",  # Replace it with your file path.
    "save_path": "log",
    "task_name": "rotate_triangle",
    "pred_horizon": 4,
    "obs_horizon": 2,
    "action_horizon": 4,
    "T_a": 4,
    "batch_size": 1,
    "num_epochs": 5000,
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "betas": [0.95, 0.999],
    "eps": 1.0e-08,
    "equiv_frac": 0.1,
    "save_freq": 500
}

# Initialize the logging path and checkpoint directory
def init_logging():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(config["save_path"], f"{config['task_name']}_train", time_str)
    os.makedirs(log_path, exist_ok=True)
    current_script = os.path.abspath(__file__)
    os.system(f"cp {current_script} {log_path}")
    checkpoint_dir = os.path.join(log_path, 'ckpts')
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

# Create the dataset and data loader
def create_dataloader():
    dataset = ToyDataset(
        dataset_path=config["dataset_path"],
        pred_horizon=config["pred_horizon"],
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"]
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )
    return train_dataloader

# Initialize the model and optimizer
def init_model_and_optimizer(device):
    noise_pred_net_in = SE3ManiNet_Invariant()
    noise_pred_net_eq = SE3ManiNet_Equivariant_Separate()
    nets = nn.ModuleDict({
        'invariant_pred_net': noise_pred_net_in,
        'equivariant_pred_net': noise_pred_net_eq
    }).to(device)
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"],
        betas=config["betas"], 
        eps=config["eps"]
    )
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=config["num_epochs"] * len(create_dataloader())
    )
    return nets, optimizer, lr_scheduler

# Prepare the input for the model
def prepare_model_input(nxyz, nrgb, noisy_actions, k, num_point):
    B = nxyz.shape[0]
    nxyz = nxyz.repeat(config["T_a"] // config["obs_horizon"], 1, 1)
    nrgb = nrgb.repeat(config["T_a"] // config["obs_horizon"], 1, 1)
    indices = [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3), (2, 0), (2, 1), (2, 3)]
    selected_elements_action = [noisy_actions[:, :, i, j] for i, j in indices]
    noisy_actions = torch.stack(selected_elements_action, dim=-1).reshape(-1, 9).unsqueeze(1).expand(-1, num_point, -1)
    tensor_k = k.clone().detach().unsqueeze(1).unsqueeze(2).expand(-1, num_point, -1)
    feature = torch.cat((nrgb, noisy_actions, tensor_k), dim=-1)
    model_input = {
        'xyz': nxyz,
        'feature': feature
    }
    return model_input

# Train a single batch of data
def train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch, device):
    nxyz = nbatch['pts'][:, :, :, :3].to(device)
    nrgb = nbatch['pts'][:, :, :, 3:].to(device)
    naction = nbatch['gt_action'].to(device)
    bz = nxyz.shape[0]
    num_point = nxyz.shape[2]
    nxyz = nxyz.view(-1, num_point, 3)
    nrgb = nrgb.view(-1, num_point, 3)
    if torch.rand(1) < config["equiv_frac"]:
        train_equiv = True
        k = torch.zeros((bz,)).long().to(device)
    else:
        train_equiv = False
        k = torch.randint(1, noise_scheduler.num_steps, (bz,), device=device)
    k = k.repeat(config["T_a"], 1).transpose(0, 1).reshape(-1)
    noisy_actions, noise = noise_scheduler.add_noise(naction, k, device=device)
    model_input = prepare_model_input(nxyz, nrgb, noisy_actions, k, num_point)
    if train_equiv:
        pred = nets["equivariant_pred_net"](model_input)
    else:
        pred = nets["invariant_pred_net"](model_input)
    noise_pred = pred
    loss, dist_r, dist_t = compute_loss(noise_pred, noise)
    if train_equiv:
        dist_equiv_r = dist_r
        dist_equiv_t = dist_t
    else:
        dist_invar_r = dist_r
        dist_invar_t = dist_t
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    loss_cpu = loss.item()
    wandb.log({"dist_R": dist_r})
    wandb.log({"dist_T": dist_t})
    wandb.log({"loss_cpu": loss_cpu})
    if train_equiv:
        wandb.log({"dist_R_eq": dist_equiv_r})
        wandb.log({"dist_T_eq": dist_equiv_t})
    else:
        wandb.log({"dist_R_in": dist_invar_r})
        wandb.log({"dist_T_in": dist_invar_t})
    return loss_cpu


# Main training function to train the model for multiple epochs
def main():
    checkpoint_dir = init_logging()
    train_dataloader = create_dataloader()
    device = torch.device('cuda')
    nets, optimizer, lr_scheduler = init_model_and_optimizer(device)
    noise_scheduler = DiffusionScheduler()
    wandb.init(
        project="SE3-Equivariant-Manipulation",
        name = time.strftime("%Y%m%d-%H%M%S"),
        notes=time.strftime("%Y%m%d-%H%M%S"),
        config={
            "learning_rate": config["learning_rate"],
            "task": config["task_name"],
            "pred_horizon": config["pred_horizon"],
            "obs_horizon": config["obs_horizon"],
            "batch_size": config["batch_size"],
            "epochs": config["num_epochs"],
            "diffusion_num_steps": noise_scheduler.num_steps,
            "diffusion_mode": noise_scheduler.mode,
            "diffusion_sigma_r": noise_scheduler.sigma_r,
            "diffusion_sigma_t": noise_scheduler.sigma_t
        }
    )
    with tqdm(range(config["num_epochs"]), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    loss_cpu = train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch, device)
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({'learning_rate': current_lr})
            wandb.log({'train_loss_avg': np.mean(epoch_loss), 'epoch': epoch_idx})
            
            if (epoch_idx + 1) % config["save_freq"] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch_idx + 1}_model.pth')
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': nets.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_cpu,
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                }, checkpoint_path)
    print("Training Done!")

if __name__ == "__main__":
    main()