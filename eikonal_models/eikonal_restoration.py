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

DEVICE = 'cuda:0'
RANDOM_SEED = 42
DATA_POINTS = 100000
TRAINING_PERCENTAGE = 0.7
SAVE_FREQUENCY = 2 # save model every i epochs
TRAIN_IDX_FOR_PLOTTING = [0, 1, 2]

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
    restoration_n_steps = args.restoration_n_steps
    batch_size = args.batch_size
    steps_per_epoch = DATA_POINTS * TRAINING_PERCENTAGE // batch_size + 1
    lr_restoration = args.lr_restoration

    restoration_checkpt_path = args.restoration_checkpt_path
    plot_path = args.plot_path

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
    print(f"Restoration model number of steps: {restoration_n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Restoration model learning rate: {lr_restoration}")
    print(f"Velocity field: {velocity_field}")
    print(f"Geometry: {geometry}")
    print(f"Linear projection x dimension (based on geometry): {linear_projection_x_dim}")
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

    # Initialize restoration model
    restoration_net = RestorationWrapper(UNet(), linear_projection_x_dim).to(DEVICE)
    restoration_net.train()
    restoration_optimizer = torch.optim.Adam(restoration_net.parameters(), lr=lr_restoration, betas=(0.9, 0.999))
    restoration_step = 0

    restoration_objective_log = []
    validation_loss_log = []
    path_to_load = f"{restoration_checkpt_path}/restoration_model_epoch_{restoration_n_steps // steps_per_epoch}.pth"

    # Set up for plotting ground truth and restored images
    stored_train_idx = TRAIN_IDX_FOR_PLOTTING
    stored_valid_idx = [train_size + i for i in TRAIN_IDX_FOR_PLOTTING]
    stored_train_y = torch.stack([transformed_dataset[idx, :, :, :] for idx in stored_train_idx]).to(DEVICE)
    stored_valid_y = torch.stack([transformed_dataset[idx, :, :, :] for idx in stored_valid_idx]).to(DEVICE)
    ground_truth_train = torch.stack([truth_dataset[idx, :, :, :] for idx in stored_train_idx]).to(DEVICE)
    ground_truth_valid = torch.stack([truth_dataset[idx, :, :, :] for idx in stored_valid_idx]).to(DEVICE)

    # plot ground truth images
    fig, axes = plt.subplots(2, len(stored_train_idx), figsize=(20, 10))
    for idx, img_idx in enumerate(stored_train_idx):
        truth_train_image = ground_truth_train[idx, 0, :, :].detach().cpu().numpy()
        plot_image(truth_train_image, axes[0, idx], f"Truth Train {img_idx}", dx, dz, srcpos, recpos)
        truth_valid_image = ground_truth_valid[idx, 0, :, :].detach().cpu().numpy()
        idx_valid = stored_valid_idx[idx]
        plot_image(truth_valid_image, axes[1, idx], f"Truth Valid {idx_valid}", dx, dz, srcpos, recpos)
    fig.savefig(f"{plot_path}/truth_img.png", dpi=100)
    plt.close(fig)

    # Load saved model checkpoint if it already exists, else run the standard training loop
    if os.path.exists(path_to_load):
        print(f"Model checkpoint exists. Loading from {path_to_load}.")
        # Setting weights_only to True to avoid FutureWarning
        checkpoint = torch.load(path_to_load, weights_only=True)
        restoration_net.load_state_dict(checkpoint['model_state_dict'])
        restoration_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restoration_step = checkpoint['epoch'] * steps_per_epoch
        restoration_objective_log = checkpoint.get('objective_log', [])
        validation_loss_log = checkpoint.get('validation_log', [])
    else:
        print("Proceeding to main restoration training loop")
        # x is from truth dataset, y is from transformed dataset
        for x, y in tqdm.tqdm(LoopLoader(train_loader, restoration_n_steps)):
            x_org = y.to(DEVICE)
            x_restored = restoration_net(x_org)
            x_truth = x.to(DEVICE)
            err = x_truth - x_restored
            objective = err.pow(2).flatten(1).mean()
            restoration_optimizer.zero_grad()
            objective.backward()
            restoration_optimizer.step()
            restoration_step += 1
            # log objective and validation loss every epoch
            if restoration_step % steps_per_epoch == 0:
                restoration_objective_log.append(objective.detach().item())
                with torch.no_grad():
                    total_val_loss = 0
                    samples = 0
                    restoration_net.eval()
                    for x_val, y_val in validation_loader:
                        # same evaluation as above, but on validation set
                        x_org_val = y_val.to(DEVICE)
                        x_restored_val = restoration_net(x_org_val)
                        x_truth_val = x_val.to(DEVICE)
                        err_val = x_truth_val - x_restored_val
                        objective_val = err_val.pow(2).flatten(1).mean()
                        total_val_loss += objective_val.item() * x_org_val.shape[0]
                        samples += x_org_val.shape[0]
                    # calculate and log the average test loss
                    avg_test_loss = total_val_loss / samples
                    validation_loss_log.append(avg_test_loss)
                    restoration_net.train()
            # Save model every steps_per_epoch * SAVE_FREQUENCY steps
            if restoration_step % (steps_per_epoch * SAVE_FREQUENCY) == 0:
                epoch_num = restoration_step // steps_per_epoch
                print(epoch_num)
                save_path = f"{restoration_checkpt_path}/restoration_model_at_epoch_{epoch_num}.pth"
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': restoration_net.state_dict(),
                    'optimizer_state_dict': restoration_optimizer.state_dict(),
                    'loss': objective.item(),
                    'objective_log': restoration_objective_log,
                    'validation_log': validation_loss_log
                }, save_path)
                print(f"Model saved at epoch {epoch_num} to {save_path}")

                with torch.no_grad():
                    restoration_net.eval()
                    x_train_restored = restoration_net(stored_train_y)
                    x_valid_restored = restoration_net(stored_valid_y)

                    # plot restored images on train and validation sets on every save
                    fig, axes = plt.subplots(2, len(stored_train_idx), figsize=(20, 10))
                    for idx, img_idx in enumerate(stored_train_idx):
                        restored_train_image = x_train_restored[idx, 0, :, :].detach().cpu().numpy()
                        plot_image(restored_train_image, axes[0, idx], f"Restored Train {img_idx}", dx, dz, srcpos, recpos)
                        restored_valid_image = x_valid_restored[idx, 0, :, :].detach().cpu().numpy()
                        idx_valid = stored_valid_idx[idx]
                        plot_image(restored_valid_image, axes[1, idx], f"Restored Valid {idx_valid}", dx, dz, srcpos, recpos)
                    fig.suptitle(f"Restored Images at Epoch {epoch_num}", fontsize=20)
                    fig.savefig(f"{restoration_checkpt_path}/restoration_img_epoch_{epoch_num}.png", dpi=300)
                    plt.close(fig)

                    restoration_net.train()


    fig = px.line(x=np.arange(len(restoration_objective_log)), y=[restoration_objective_log, validation_loss_log], labels={'x': 'Epochs', 'y': 'Loss'})
    fig.data[0].name = "Train Loss"
    fig.data[1].name = "Validation Loss"
    fig.update_layout(
        height=400, 
        width=550,
        yaxis=dict(
            type='log',
            title='Loss'
        ),
        title="Restoration Model"
    )

    pio.write_html(fig, f'{plot_path}/restoration_loss.html')

    restoration_epoch_num = restoration_n_steps // steps_per_epoch
    restoration_path_to_load = f"{restoration_checkpt_path}/restoration_model_epoch_{restoration_epoch_num}.pth"
    # Load saved model checkpoint if it already exists, else run the standard training loop
    if os.path.exists(restoration_path_to_load):
        print(f"Model checkpoint exists. Loading from {restoration_path_to_load}.")
        # Setting weights_only to True to avoid FutureWarning
        checkpoint = torch.load(restoration_path_to_load, weights_only=True)
        restoration_net.load_state_dict(checkpoint['model_state_dict'])
        restoration_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restoration_step = checkpoint['epoch'] * steps_per_epoch
        restoration_objective_log = checkpoint.get('objective_log', [])
        validation_loss_log = checkpoint.get('validation_log', [])
    else:
        raise Exception("Restoration model checkpoint does not exist. Please check the path.")

    # Plot restored image vs ground truth for train and test sets
    plot_restoration_model_images(train_loader, restoration_net, dx, dz, srcpos, recpos, plots_dir=plot_path, device=DEVICE, batch=20)
    plot_restoration_model_images(test_loader, restoration_net, dx, dz, srcpos, recpos, plots_dir=plot_path, device=DEVICE, batch=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--restoration-n-steps", type=int, required=True, help="Number of steps for running restoration model")
    parser.add_argument("--batch_size", type=int, required=True, help="Eikonal data batch size")
    parser.add_argument("--lr-restoration", type=float, required=True, help="Restoration model learning rate")
    # Dataset parameters
    parser.add_argument("--dataset-type", type=str, required=True, help="Type of eikonal dataset (i.e. velocity field) used - can either be 'fixed_grad' or 'grad'")
    parser.add_argument("--geometry", type=str, required=True, help="Type of velocity geometry - can either be 'rand' or 'transmission'")
    # Model/plotting paths
    parser.add_argument("--restoration-checkpt-path", type=str, required=True, help="Path for saved restoration model checkpoints")
    parser.add_argument("--plot-path", type=str, required=True, help="Path for all plots")

    args = parser.parse_args()
    main(args)