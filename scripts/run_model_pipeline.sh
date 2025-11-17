#!/bin/bash
# Model parameters
restoration_n_steps=2800
nppc_n_steps=1500
batch_size=256
lr_restoration=5e-4
lr_nppc=1e-4

# Select Dataset and Geometry
echo "Select the dataset type:"
select dataset_type in "fixed_grad" "grad"; do
  if [[ -n "$dataset_type" ]]; then
    echo "You selected: $dataset_type"
    break
  else
    echo "Invalid selection. Please try again."
  fi
done

echo "Select the geometry type:"
select dataset_geometry in "transmission" "rand"; do
  if [[ -n "$dataset_geometry" ]]; then
    echo "You selected: $dataset_geometry"
    break
  else
    echo "Invalid selection. Please try again."
  fi
done

# Model checkpoint and plotting paths
home_path="/eikonal_results/$dataset_type/$dataset_geometry"
restoration_checkpt_path="$home_path/restoration_checkpoints"
nppc_checkpt_path="$home_path/nppc_checkpoints"
plots_path="$home_path/plots"
# Path to restoration model checkpoint, 
restoration_net_path="$restoration_checkpt_path/restoration_model_at_epoch_10.0.pth"

# Check and create storage paths if they don't exist
if [ ! -d "$restoration_checkpt_path" ]; then
    echo "Restoration checkpoint path does not exist. Creating: $restoration_checkpt_path"
    mkdir -p "$restoration_checkpt_path"
else
    echo "Restoration checkpoint path exists: $restoration_checkpt_path"
fi

if [ ! -d "$nppc_checkpt_path" ]; then
    echo "NPPC checkpoint path does not exist. Creating: $nppc_checkpt_path"
    mkdir -p "$nppc_checkpt_path"
else
    echo "NPPC checkpoint path exists: $nppc_checkpt_path"
fi

if [ ! -d "$plots_path" ]; then
    echo "Plots path does not exist. Creating: $plots_path"
    mkdir -p "$plots_path"
else
    echo "Plots path exists: $plots_path"
fi

# Check if restoration model checkpoint exists
if [ ! -f "$restoration_net_path" ]; then
    echo "Warning: File $restoration_net_path does not exist. You should run the restoration model first."
fi

# Additional nppc parameters
second_moment_loss_lambda=1e0
second_moment_loss_grace=500

# Run restoration model 
python3 ../eikonal_models/eikonal_restoration.py \
    --restoration-n-steps "$restoration_n_steps" \
    --batch_size "$batch_size" \
    --lr-restoration "$lr_restoration" \
    --dataset-type "$dataset_type" \
    --geometry "$dataset_geometry" \
    --restoration-checkpt-path "$restoration_checkpt_path" \
    --plot-path "$plots_path"

# Run NPPC
python3 ../eikonal_models/eikonal_nppc.py \
    --nppc-n-steps "$nppc_n_steps" \
    --restoration-n-steps "$restoration_n_steps" \
    --batch_size "$batch_size" \
    --lr-nppc "$lr_nppc" \
    --lr-restoration "$lr_restoration" \
    --second-moment-loss-lambda "$second_moment_loss_lambda" \
    --second-moment-loss-grace "$second_moment_loss_grace" \
    --dataset-type "$dataset_type" \
    --geometry "$dataset_geometry" \
    --nppc-checkpt-path "$nppc_checkpt_path" \
    --restoration-checkpt-path "$restoration_checkpt_path" \
    --plot-path "$plots_path" \
    --restoration-net-path "$restoration_net_path"
