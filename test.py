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


# Configure parameters
config = {
    "dataset_path": "data_path.npy", # replace with your data path
    "save_path": "log",
    "task_name": "rotate_triangle",
    "pred_horizon": 4,
    "obs_horizon": 1,
    "action_horizon": 4,
    "T_a": 4,
    "batch_size": 1,
    "checkpoint_path": "",  # replace with your checkpoint path
}

# Create the dataset and data loader
def create_dataloader():
    dataset = ToyDataset(
        dataset_path=config["dataset_path"],
        pred_horizon=config["pred_horizon"],
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader

# Initialize the model and optimizer
def init_model(device):
    noise_pred_net_in = SE3ManiNet_Invariant()
    noise_pred_net_eq = SE3ManiNet_Equivariant_Separate()
    nets = nn.ModuleDict({
        'invariant_pred_net': noise_pred_net_in,
        'equivariant_pred_net': noise_pred_net_eq
    }).to(device)
    checkpoint = torch.load(config["checkpoint_path"])
    nets.load_state_dict(checkpoint['model_state_dict'])
    nets.eval()
    return nets

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


# test a single batch of data
def test_batch(nets, noise_scheduler, nbatch, device):
    nets.eval()
    with torch.no_grad():
        nxyz = nbatch['pts'][:, :, :, :3].to(device)
        nrgb = nbatch['pts'][:, :, :, 3:].to(device)
        naction = nbatch['gt_action'].to(device)
        bz = nxyz.shape[0]
        num_point = nxyz.shape[2]
        nxyz = nxyz.view(-1, num_point, 3)
        nrgb = nrgb.view(-1, num_point, 3)
        
        H_t_noise = torch.eye(4).expand(config["action_horizon"], 4, 4).unsqueeze(0).to(device) # noise action initialize
        
        kk = torch.zeros((bz,)).long().to(device)
        kk = kk.repeat(config["T_a"], 1).transpose(0, 1).reshape(-1)
        kk[:] = noise_scheduler.num_steps - 1 
        for denoise_idx in range(noise_scheduler.num_steps - 1, -1, -1):
            k = torch.zeros((bz,)).long().to(device)
            k = k.repeat(config["T_a"], 1).transpose(0, 1).reshape(-1)
            k[:] = denoise_idx
            model_input = prepare_model_input(nxyz, nrgb, H_t_noise, k, num_point)
            
            if (denoise_idx == 0): 
                test_equiv = True 
            else: 
                test_equiv = False
            
            with torch.no_grad():
                if (test_equiv):
                    pred = nets["equivariant_pred_net"](model_input)
                else:
                    pred = nets["invariant_pred_net"](model_input)
    
            noise_pred = pred
            H_t_noise, H_0 = noise_scheduler.denoise(
                model_output = noise_pred,
                timestep = k,
                sample = H_t_noise,
                device = device
            )
            
        loss, dist_R, dist_T = compute_loss(H_0.squeeze(0), naction.squeeze(0))
        # print("loss: ", loss)
        loss_cpu = loss.item()
        if test_equiv:
            dist_equiv_r = dist_R
            dist_equiv_t = dist_T
        else:
            dist_invar_r = dist_R
            dist_invar_t = dist_T
        wandb.log({"test_dist_R": dist_R})
        wandb.log({"test_dist_T": dist_T})
        wandb.log({"test_loss_cpu": loss_cpu})
        if test_equiv:
            wandb.log({"test_dist_R_eq": dist_equiv_r})
            wandb.log({"test_dist_T_eq": dist_equiv_t})
        else:
            wandb.log({"test_dist_R_in": dist_invar_r})
            wandb.log({"test_dist_T_in": dist_invar_t})
    return loss_cpu


# Main function
def main():
    dataloader = create_dataloader()
    device = torch.device('cuda')
    nets = init_model(device)
    noise_scheduler = DiffusionScheduler()
    wandb.init(
        project="SE3-Equivariant-Manipulation",
        notes=time.strftime("%Y%m%d-%H%M%S"),
        config={
            "task": config["task_name"],
            "pred_horizon": config["pred_horizon"],
            "obs_horizon": config["obs_horizon"],
            "batch_size": config["batch_size"],
            "diffusion_num_steps": noise_scheduler.num_steps,
            "diffusion_mode": noise_scheduler.mode,
            "diffusion_sigma_r": noise_scheduler.sigma_r,
            "diffusion_sigma_t": noise_scheduler.sigma_t
        }
    )
    test_losses = []
    with tqdm(dataloader, desc='Test Batch') as tepoch:
        for nbatch in tepoch:
            loss_cpu = test_batch(nets, noise_scheduler, nbatch, device)
            test_losses.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)
    avg_test_loss = np.mean(test_losses)
    wandb.log({'test_loss': avg_test_loss})
    print(f"Test Done! Average Test Loss: {avg_test_loss}")

if __name__ == "__main__":
    main()
