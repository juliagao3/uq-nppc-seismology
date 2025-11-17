import argparse
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

import os
import tqdm
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

from IPython.display import Markdown
from models import *
from plot_util import *

N_DIRS = 5
DEVICE = 'cuda:0'
RANDOM_SEED = 42
DATA_POINTS = 100000
TRAINING_PERCENTAGE = 0.7
SAVE_FREQUENCY = 2 # save model every i epochs

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.benchmark = True

class LoopLoader():
    def __init__(self, dataloader, size):
        self.dataloader = dataloader
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        i = 0
        while (i < self.size):
            for x in self.dataloader:
                if (i >= self.size):
                    break
                yield x
                i += 1


def main(args):
    # Obtain necessary model, data, and path parameters
    nppc_n_steps = args.nppc_n_steps
    restoration_n_steps = args.restoration_n_steps
    batch_size = args.batch_size
    lr_nppc = args.lr_nppc
    lr_restoration = args.lr_restoration
    
    second_moment_loss_lambda = args.second_moment_loss_lambda
    second_moment_loss_grace = args.second_moment_loss_grace

    steps_per_epoch = DATA_POINTS * TRAINING_PERCENTAGE // batch_size + 1

    nppc_checkpt_path = args.nppc_checkpt_path
    restoration_checkpt_path = args.restoration_checkpt_path
    plot_path = args.plot_path
    restoration_net_path = args.restoration_net_path

    # Load corresponding velocity field
    # There are only two types of datasets: FixedGrad and Grad
    if args.dataset_type == "fixed_grad":
        velocity_field = "0.3FixedGradGRFSamples100000_28x28_1_1.5"
    else:
        velocity_field = "GradGRFSamples100000_28x28_1_1.5"
    
    geometry = args.geometry
    # There are only two types of geometries: random ('rand') and transmission ('transmission')
    if geometry == "rand":
        # In random geometry, datapoints are of size 10 x 26
        linear_projection_x_dim = 10
    else:
        # In transmission geometry, datapoints are of size 26 x 26
        linear_projection_x_dim = 26

    print("Printing important variables...")
    print(f"NPPC number of steps: {nppc_n_steps}")
    print(f"Restoration model number of steps: {restoration_n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"NPPC learning rate: {lr_nppc}")
    print(f"Restoration model learning rate: {lr_restoration}")
    print(f"Second moment loss lambda: {second_moment_loss_lambda}")
    print(f"Second moment loss grace: {second_moment_loss_grace}")
    print(f"Velocity field: {velocity_field}")
    print(f"Geometry: {geometry}")
    print(f"Linear projection x dimension (based on geometry): {linear_projection_x_dim}")
    print(f"NPPC checkpoint path: {nppc_checkpt_path}")
    print(f"Restoration checkpoint path: {restoration_checkpt_path}")
    print(f"Plots path: {plot_path}")

    recpos = np.load("/groups/mlprojects/cs163-seismology/data_for_cs163/{}_{}/RecPos.npy".format(velocity_field, geometry))
    srcpos = np.load("/groups/mlprojects/cs163-seismology/data_for_cs163/{}_{}/SouPos.npy".format(velocity_field, geometry))
    vel = np.load("/groups/mlprojects/cs163-seismology/data_for_cs163/{}.npy".format(velocity_field))  # km / s

    dx = 0.25  # km
    dz = 0.25  # km

    nx = vel.shape[0]
    nz = vel.shape[1]
    xcoor = np.arange(nx) * dx
    zcoor = np.arange(nz) * dz
    xx, zz = np.meshgrid(xcoor, zcoor, indexing='ij')
    srcx, srcz = xcoor[srcpos[:, 0]], zcoor[srcpos[:, 1]]
    recx, recz = xcoor[recpos[:, 0]], zcoor[recpos[:, 1]]

    travel_dataset = np.load("/groups/mlprojects/cs163-seismology/data_for_cs163/{}_{}/TT_0_to_100000.npy".format(velocity_field, geometry))
    vel_dataset = np.load("/groups/mlprojects/cs163-seismology/data_for_cs163/{}.npy".format(velocity_field))

    #transpose dataset
    travel_dataset = travel_dataset.transpose(2, 0, 1)
    travel_dataset = torch.tensor(travel_dataset)
    transformed_dataset = travel_dataset.unsqueeze(1).repeat(1, 1, 1, 1)

    vel_dataset = vel_dataset.transpose(2, 0, 1)
    vel_dataset = torch.tensor(vel_dataset)
    truth_dataset = vel_dataset.unsqueeze(1).repeat(1, 1, 1, 1)

    # Train/Validation/Test is roughly 70%/15%/15% split
    train_size = int(0.7 * transformed_dataset.shape[0])
    validation_size = int((len(transformed_dataset) - train_size) / 2)
    test_size = int((len(transformed_dataset) - train_size) / 2)

    truth_dataset = truth_dataset[:(test_size + validation_size + train_size), :, :, :]

    # Construct train, validation, and test datasets
    train_dataset = torch.utils.data.TensorDataset(truth_dataset[:train_size], transformed_dataset[:train_size])
    validation_dataset = torch.utils.data.TensorDataset(truth_dataset[train_size:(train_size + validation_size)], 
                                                        transformed_dataset[train_size:(train_size + validation_size)])
    test_dataset = torch.utils.data.TensorDataset(truth_dataset[(train_size + validation_size):(train_size+validation_size+test_size)]
                                                , transformed_dataset[(train_size + validation_size):(train_size+validation_size+test_size)])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Resume the restoration model
    restoration_checkpoint = torch.load(restoration_net_path)
    restoration_net = RestorationWrapper(UNet(), linear_projection_x_dim).to(DEVICE)
    restoration_net.load_state_dict(restoration_checkpoint['model_state_dict'])
    restoration_optimizer = torch.optim.Adam(restoration_net.parameters(), lr=lr_restoration, betas=(0.9, 0.999))
    restoration_optimizer.load_state_dict(restoration_checkpoint['optimizer_state_dict'])
    restoration_net.eval()

    # Initialize nppc model
    nppc_net = PCWrapper(UNet(in_channels=1 + 1, out_channels=1 * N_DIRS), n_dirs=N_DIRS, linear_projection_x_dim=linear_projection_x_dim).to(DEVICE)
    nppc_net.train()
    nppc_optimizer = torch.optim.Adam(nppc_net.parameters(), lr=lr_nppc, betas=(0.9, 0.999))
    nppc_step = 0 

    # logs
    nppc_objective_log = []
    validation_loss_log = []
    path_to_load = f"{nppc_checkpt_path}/nppc_model_epoch_{nppc_n_steps // steps_per_epoch}.pth"
    # Load saved model nppc checkpoint if it exists, else run the standard training loop
    if os.path.exists(path_to_load):
        print(f"Model checkpoint exists. Loading from {path_to_load}.")
        # Setting weights_only to True to avoid FutureWarning
        checkpoint = torch.load(path_to_load, weights_only=True)
        nppc_net.load_state_dict(checkpoint['model_state_dict'])
        nppc_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        nppc_step = checkpoint['epoch'] * steps_per_epoch
        nppc_objective_log = checkpoint.get('objective_log', [])
        validation_loss_log = checkpoint.get('validation_log', [])
    else:
        print("Proceeding to main NPPC training loop.")
        for x, y in tqdm.tqdm(LoopLoader(train_loader, nppc_n_steps)):
            x_org = y.to(DEVICE)
            with torch.no_grad():
                x_restored = restoration_net(x_org)
            x_truth = x.to(DEVICE)
            err = x_truth - x_restored
            w_mat = nppc_net(x_org, x_restored)
            w_mat_ = w_mat.flatten(2)
            w_norms = w_mat_.norm(dim=2)
            w_hat_mat = w_mat_ / w_norms[:, :, None].double()
            err = (x_truth - x_restored).flatten(1)
            ## Normalizing by the error's norm
            ## -------------------------------
            err_norm = err.norm(dim=1)
            err = err / err_norm[:, None]
            w_norms = w_norms / err_norm[:, None]
            ## W hat loss
            ## ----------
            err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
            reconst_err = 1 - err_proj.pow(2).sum(dim=1)
            ## W norms loss
            ## ------------
            second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
            second_moment_loss_lambda = -1 + 2 * nppc_step / second_moment_loss_grace
            second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1) ,1e-6)
            second_moment_loss_lambda *= second_moment_loss_lambda
            objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
            nppc_optimizer.zero_grad()
            objective.backward()
            nppc_optimizer.step()
            nppc_step += 1
            if nppc_step % steps_per_epoch:
                nppc_objective_log.append(objective.detach().item())
                with torch.no_grad():
                    test_val_loss = 0
                    samples = 0
                    nppc_net.eval()
                    for x_val, y_val in validation_loader:
                        x_org_val = y_val.to(DEVICE)
                        x_restored_val = restoration_net(x_org_val)
                        x_truth_val = x_val.to(DEVICE)
                        err = x_truth_val - x_restored_val
                        w_mat = nppc_net(x_org_val, x_restored_val)
                        w_mat_ = w_mat.flatten(2)
                        w_norms = w_mat_.norm(dim=2)
                        w_hat_mat = w_mat_ / w_norms[:, :, None].double()
                        err = (x_truth_val - x_restored_val).flatten(1)
                        ## Normalizing by the error's norm
                        ## -------------------------------
                        err_norm = err.norm(dim=1)
                        err = err / err_norm[:, None]
                        w_norms = w_norms / err_norm[:, None]
                        ## W hat loss
                        ## ----------
                        err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
                        reconst_err = 1 - err_proj.pow(2).sum(dim=1)
                        ## W norms loss
                        ## ------------
                        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
                        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
                        test_val_loss += objective.item() * x_org_val.shape[0]
                        samples += x_org_val.shape[0]
                            # Average test loss
                    avg_test_loss = test_val_loss / samples
                    validation_loss_log.append(avg_test_loss)
                    nppc_net.train()
            # Save model every 10 epochs
            if nppc_step % (steps_per_epoch * SAVE_FREQUENCY) == 0:
                save_path = f"{nppc_checkpt_path}/nppc_model_epoch_{nppc_step // steps_per_epoch}.pth"
                torch.save({
                    'epoch': nppc_step // steps_per_epoch,
                    'model_state_dict': nppc_net.state_dict(),
                    'optimizer_state_dict': nppc_optimizer.state_dict(),
                    'loss': objective.item(),
                    'objective_log': nppc_objective_log,
                    'validation_log': validation_loss_log
                }, save_path)
                print(f"Model saved at step {nppc_step} to {save_path}")


    fig = px.line(x=np.arange(len(nppc_objective_log)), y=[nppc_objective_log, validation_loss_log], labels={'x': 'Epochs', 'y': 'Loss'})
    fig.data[0].name = "Train Loss"
    fig.data[1].name = "Validation Loss"
    fig.update_layout(
        height=400, 
        width=550,
        yaxis=dict(
            type='log',
            title='Loss'
        ),
        title="NPPC Model"
    )

    pio.write_html(fig, f'{plot_path}/nppc_loss.html')
    restoration_epoch_num = restoration_n_steps // steps_per_epoch
    nppc_epoch_num = nppc_n_steps // steps_per_epoch

    # Restoration
    restoration_path_to_load = f"{restoration_checkpt_path}/restoration_model_epoch_{restoration_epoch_num}.pth"
    # Load saved model checkpoint if it already exists, else run the standard training loop
    if os.path.exists(restoration_path_to_load):
        print(f"Model checkpoint exists. Loading from {restoration_path_to_load}.")
        # Setting weights_only to True to avoid FutureWarning
        checkpoint = torch.load(restoration_path_to_load, weights_only=True)
        restoration_optimizer = checkpoint['optimizer']
        validation_loss_log = checkpoint.get('validation_log', [])

    # NPPC
    nppc_path_to_load = f"{nppc_checkpt_path}/nppc_model_epoch_{nppc_epoch_num}.pth"
    # Load saved model checkpoint if it exists, else run the standard training loop
    if os.path.exists(nppc_path_to_load):
        print(f"Model checkpoint exists. Loading from {nppc_path_to_load}.")
        # Setting weights_only to True to avoid FutureWarning
        checkpoint = torch.load(nppc_path_to_load, weights_only=True)
        nppc_net.load_state_dict(checkpoint['model_state_dict'])
        nppc_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        nppc_step = checkpoint['epoch'] * steps_per_epoch
        nppc_objective_log = checkpoint.get('objective_log', [])
        validation_loss_log = checkpoint.get('validation_log', [])
    
    plot_nppc(train_loader, restoration_net, nppc_net, dx, dz, srcpos, recpos, plots_dir =plot_path, device=DEVICE)
    plot_nppc(test_loader, restoration_net, nppc_net, dx, dz, srcpos, recpos, plots_dir=plot_path, device=DEVICE)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--nppc-n-steps", type=int, required=True, help="Number of steps for running NPPC")
    parser.add_argument("--restoration-n-steps", type=int, required=True, help="Number of steps for running restoration model")
    parser.add_argument("--batch_size", type=int, required=True, help="Eikonal data batch size")
    parser.add_argument("--lr-nppc", type=float, required=True, help="NPPC model learning rate")
    parser.add_argument("--lr-restoration", type=float, required=True, help="Restoration model learning rate")
    # NPPC-specific parameters
    parser.add_argument("--second-moment-loss-lambda", type=float, required=True, help="Second moment loss lambda (NPPC only)")
    parser.add_argument("--second-moment-loss-grace", type=float, required=True, help="Second moment loss grace (NPPC only)")
    # Dataset parameters
    parser.add_argument("--dataset-type", type=str, required=True, help="Type of eikonal dataset (i.e. velocity field) used - can either be 'fixed_grad' or 'grad'")
    parser.add_argument("--geometry", type=str, required=True, help="Type of velocity geometry - can either be 'rand' or 'transmission'")
    # Model/plotting paths
    parser.add_argument("--nppc-checkpt-path", type=str, required=True, help="Path for saved NPPC checkpoints")
    parser.add_argument("--restoration-checkpt-path", type=str, required=True, help="Path for saved restoration model checkpoints")
    parser.add_argument("--plot-path", type=str, required=True, help="Path for all plots")
    parser.add_argument("--restoration-net-path", type=str, required=True, help="Path with saved restoration model")

    args = parser.parse_args()
    main(args)