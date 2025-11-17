import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from ipywidgets import interactive
from IPython.display import display

def plot_restoration_row(images, axes, row_index, title_prefix, dx, dz, srcpos, recpos, batch=20):
    """
    Plot a row of images for restoration model for a batch. Source and receiver markers are annotated
    on the plot.
    """
    for i in range(batch):
        image = images[i, 0, :, :].cpu().numpy()

        nx, nz = image.shape
        xcoor = np.arange(nx) * dx
        zcoor = np.arange(nz) * dz
        srcx, srcz = xcoor[srcpos[:, 0]], zcoor[srcpos[:, 1]]
        recx, recz = xcoor[recpos[:, 0]], zcoor[recpos[:, 1]]

        xx, zz = np.meshgrid(xcoor, zcoor, indexing='ij')

        ax = axes[row_index, i]
        vim = ax.pcolormesh(xx, zz, image, cmap='jet_r', vmin=3.5, vmax=6.5)
        cbar = plt.colorbar(vim, ax=ax)
        ax.plot(srcx, srcz, '.', color='r', label='source', markersize=20)
        ax.plot(recx, recz, 'v', color='b', label='receiver', markersize=20)
        ax.invert_yaxis()
        ax.set_xlabel('Distance (km)', fontsize=50) 
        ax.set_ylabel('Depth (km)', fontsize=50)
        ax.grid(color='gray', linestyle='-', linewidth=0.7, alpha=0.8)
        ax.set_title(f"{title_prefix} {i+1}", fontsize=30)


def plot_restoration_model_images(dataloader, restoration_net, dx, dz, srcpos, recpos, plots_dir, device, batch=20):
    """
    Plot restoration model images (restored vs. truth) for a given dataset.
    """
    # Load data and run restoration model
    x_org, y_org = next(iter(dataloader))
    with torch.no_grad():
        x_restored = restoration_net(y_org.to(device))
    x_truth = x_org.to(device)
    err = x_truth - x_restored
    print("Error for this batch:", (err.pow(2).flatten(1).mean()).item())

    fig, axes = plt.subplots(2, batch, figsize=(200, 20))

    # Plot restored images
    plot_restoration_row(x_restored, 
             axes, 
             row_index=0, 
             title_prefix="Restored Image", 
             dx=dx, 
             dz=dz, 
             srcpos=srcpos, 
             recpos=recpos, 
             batch=20)
    
    # Plot ground truth images
    plot_restoration_row(x_org, 
             axes, 
             row_index=1, 
             title_prefix="Ground Truth Image", 
             dx=dx, 
             dz=dz, 
             srcpos=srcpos, 
             recpos=recpos, 
             batch=20)

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/restoration_imgs.png", dpi=200)


def plot_nppc_row(images, axes, dx, dz, srcpos, recpos, title_prefix=""):
    """
    Plot a row of images for NPPC for a batch. Source and receiver markers are annotated
    on the plot.
    """
    
    for i in range(6):
        image = images[i, 0, :, :].cpu().numpy()
        nx, nz = image.shape
        xcoor = np.arange(nx) * dx
        zcoor = np.arange(nz) * dz
        srcx, srcz = xcoor[srcpos[:, 0]], zcoor[srcpos[:, 1]]
        recx, recz = xcoor[recpos[:, 0]], zcoor[recpos[:, 1]]
        xx, zz = np.meshgrid(xcoor, zcoor, indexing='ij')

        ax = axes[i]
        vim = ax.pcolormesh(xx, zz, image, cmap='jet_r', vmin=3.5, vmax=6.5)
        cbar = plt.colorbar(vim, ax=ax)
        ax.plot(srcx, srcz, '.', color='r', label='source', markersize=20)
        ax.plot(recx, recz, 'v', color='b', label='receiver', markersize=20)
        ax.invert_yaxis()
        ax.set_xlabel('Distance (km)', fontsize=50)  
        ax.set_ylabel('Depth (km)', fontsize=50)
        ax.grid(color='gray', linestyle='-', linewidth=0.7, alpha=0.8)
        ax.set_title(f"{title_prefix} {1}", fontsize=30)


def plot_nppc(dataloader, restoration_net, nppc_net, dx, dz, srcpos, recpos, plots_dir, device):
    """
    Plot NPPC results: ground truth, restored images, and principal components.
    """
    # Load data and compute restoration and NPPC
    x_org, y_org = next(iter(dataloader))
    with torch.no_grad():
        x_restored = restoration_net(y_org.to(device))
        w_mat = nppc_net(y_org.to(device), x_restored.to(device))
    x_truth = x_org.to(device)
    err = x_truth - x_restored
    print("Error for this batch:", (err.pow(2).flatten(1).mean()).item())

    fig, axes = plt.subplots(1, 6, figsize=(40, 30))
    # Plot restored image(s)
    plot_nppc_row(x_restored, axes, dx=dx, dz=dz, srcpos=srcpos, recpos=recpos,
                  title_prefix="Restored Image")
    # Plot ground truth image(s)
    plot_nppc_row(x_org, axes, dx=dx, dz=dz, srcpos=srcpos, recpos=recpos,
                  title_prefix="Ground Truth Image")

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/nppc_truth_restored.png", dpi=200)

    # Plot principal components and uncertainties
    fig, axes = plt.subplots(6, 6, figsize=(38, 35))

    for i in range(6):
        for j in range(6):
            ax = axes[i, j]  

            # Select the specific image for this subplot
            if j == 0:  # For the first column, use the same image (first image)
                restored_image = x_restored[i, 0, :, :].cpu().numpy()

                nx = restored_image.shape[0]
                nz = restored_image.shape[1]
                
                xcoor = np.arange(nx) * dx
                zcoor = np.arange(nz) * dz
                srcx, srcz = xcoor[srcpos[:, 0]], zcoor[srcpos[:, 1]]
                recx, recz = xcoor[recpos[:, 0]], zcoor[recpos[:, 1]]
                
                xx, zz = np.meshgrid(xcoor, zcoor, indexing='ij')
                
                vim = ax.pcolormesh(xx, zz, restored_image, cmap='jet_r', vmin=3.5, vmax=6.5)
                cbar = plt.colorbar(vim, ax=ax) 
                ax.plot(srcx, srcz, '.', color='r', label='source', markersize=15)
                ax.plot(recx, recz, 'v', color='b', label='receiver', markersize=15)
                ax.invert_yaxis()
                ax.set_xlabel('Distance (km)', fontsize=25) 
                ax.set_ylabel('Depth (km)', fontsize=25)
                ax.set_title(f"Restored Image", fontsize=30)
            else:  # For the rest of the columns, use a different image 
                wmat_image =  w_mat[i, j-1, 0, :, :].detach().cpu().numpy()
                nx, nz = wmat_image.shape
            
                cmap = plt.get_cmap('RdBu_r')
                ax.set_aspect(1)
                
                vim1 = ax.pcolormesh(xx, zz, wmat_image, cmap='seismic', vmin=-0.1, vmax=0.1)
                cbar1 = plt.colorbar(vim1, ax=ax)
                
                ax.invert_yaxis() 
                ax.set_title(f"Principal components {j}", fontsize=25)
            
    plt.tight_layout()
    fig.savefig(f"{plots_dir}/nppc_principal_components.png", dpi=200)

def plot_train_over_time_restoration(dataloader, restoration_net, dx, dz, srcpos, recpos, plots_dir, device, fixed_indexes=[0, 1, 2]):
    """
    Plot restored images from the restoration model over time using fixed indexes from the dataset.
    This function visualizes the changes in the restoration process for a batch as training progresses.
    """
    x_org, y_org = next(iter(dataloader))

    x_org = x_org.to(device)
    y_org = y_org.to(device)

    with torch.no_grad():
        x_restored = restoration_net(y_org)

    num_images = len(fixed_indexes)
    fig, axes = plt.subplots(2, num_images, figsize=(5 * num_images, 10))

    # plot restored images and ground truth for fixed indexes
    for idx, fixed_index in enumerate(fixed_indexes):
        # ground truth
        plot_image(x_org[fixed_index], axes[0, idx], title=f"Ground Truth {idx + 1}", dx=dx, dz=dz, srcpos=srcpos, recpos=recpos)

        # restored image
        plot_image(x_restored[fixed_index], axes[1, idx], title=f"Restored {idx + 1}", dx=dx, dz=dz, srcpos=srcpos, recpos=recpos)

    axes[0, 0].set_ylabel("Ground Truth", fontsize=20)
    axes[1, 0].set_ylabel("Restored", fontsize=20)
    plt.tight_layout()

    save_path = f"{plots_dir}/restoration_train_progress.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved training visualization to {save_path}")

def plot_image(image, ax, title, dx, dz, srcpos, recpos):
    """
    Helper function to plot an individual image with annotations.
    """
    nx, nz = image.shape
    xcoor = np.arange(nx) * dx
    zcoor = np.arange(nz) * dz
    srcx, srcz = xcoor[srcpos[:, 0]], zcoor[srcpos[:, 1]]
    recx, recz = xcoor[recpos[:, 0]], zcoor[recpos[:, 1]]
    xx, zz = np.meshgrid(xcoor, zcoor, indexing='ij')
    vim = ax.pcolormesh(xx, zz, image, cmap='jet_r', vmin=3.5, vmax=6.5)
    plt.colorbar(vim, ax=ax)
    ax.plot(srcx, srcz, '.', color='r', label='source')
    ax.plot(recx, recz, 'v', color='b', label='receiver')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=20)

def plot_image_batch(image_tensor, ax, title, dx, dz, srcpos, recpos):
    """
    Helper function to plot a single image with source and receiver markers.

    Args:
        image_tensor: Image tensor to plot (CxHxW format).
        ax
        title
        dx, dz: Grid spacing in x and z directions.
        srcpos, recpos: Source and receiver positions.
    """
    image = image_tensor[0, :, :].cpu().numpy()

    nx, nz = image.shape
    xcoor = np.arange(nx) * dx
    zcoor = np.arange(nz) * dz
    srcx, srcz = xcoor[srcpos[:, 0]], zcoor[srcpos[:, 1]]
    recx, recz = xcoor[recpos[:, 0]], zcoor[recpos[:, 1]]
    xx, zz = np.meshgrid(xcoor, zcoor, indexing='ij')

    vim = ax.pcolormesh(xx, zz, image, cmap='jet_r', vmin=3.5, vmax=6.5)
    plt.colorbar(vim, ax=ax)
    ax.plot(srcx, srcz, '.', color='r', label='source')
    ax.plot(recx, recz, 'v', color='b', label='receiver')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=15)


def create_uncertainties_gif(w_mat, batch_index, x_restored, alphas=torch.linspace(-5, 5, steps=20, device="cpu")):
    """
    Visualize the effect of linearly interpolating `w_mat` on `x_restored`.

    Args:
        w_mat (torch.Tensor): output of `restored_model.get_dirs(x_distorted, x_restored)`.
        batch_index (int): index of the image to visualize.
        x_restored (torch.Tensor): mean posterior of the restored image.
        alphas (torch.Tensor): range of alpha values for interpolation (default: linspace[-5, 5, 20]).

    Returns:
        None

    Example usage:
        visualize_interpolation(w_first_five, batch_index, x_restored, alphas)
    """
    # example usage:
    # visualize_interpolation(w_first_five, batch_index, x_restored, alphas)

    def update(alpha_index):
        alpha = alphas[alpha_index]
        x_alpha = x_restored + alpha * w_mat[batch_index]
        x_alpha_clamped = torch.clamp(x_alpha, 0, 1)  # ensure values are in [0, 1]

        x_frame = make_grid(x_alpha_clamped, nrow=1).permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(x_frame)
        plt.axis('off')
        plt.title(f"Alpha: {alpha.item():.2f}")
        plt.show()

    # create interactive slider
    slider = interactive(update, alpha_index=(0, len(alphas) - 1))
    display(slider)

