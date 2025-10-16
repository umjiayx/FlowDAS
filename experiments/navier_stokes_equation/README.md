# FlowDAS: Flow-based Data Assimilation for Navier-Stokes Equations

A deep learning framework for forecasting and data assimilation in fluid dynamics using stochastic interpolants and flow matching.

## Installation

Please see the main page for installing the environment. 

## Quick Start

### Training

Train a model from scratch:

```bash
python main.py --config configs/config_training.yaml
```

Key training parameters in `configs/config_training.yaml`:
- `time_window`: Number of previous frames for conditioning (default: 10)
- `auto_step`: Number of future frames to predict (default: 1)
- `batch_size`: Training batch size
- `max_steps`: Total training iterations

### Inference

Run inference with a trained model:

```bash
python main.py --config configs/config_inference.yaml
```

**Select specific trajectory and starting time:**

```bash
python main.py --config configs/config_inference.yaml \
    --trajectory_index 2 \
    --time_index 20
```

**Test on multiple trajectories:**

```bash
for i in {0..9}; do
    python main.py --config configs/config_inference.yaml \
        --trajectory_index $i \
        --savedir results/traj_$i
done
```

## Configuration

### Key Parameters

**Model Architecture** (`model` section):
- `channels`: Base UNet channels (default: 128)
- `dim_mults`: Channel multipliers at each resolution level

**FlowDAS** (`flowdas` section):
- `time_window`: Historical frames for conditioning
- `auto_step`: Future frames to predict
- `grad_scale`: Gradient scaling for data assimilation (default: 1.0)

**Sampling** (`sampling` section):
- `EM_sample_steps`: Euler-Maruyama sampling steps
- `trajectory_index`: Which trajectory to use for inference
- `time_index`: Starting time in the trajectory
- `load_path`: Path to checkpoint file

**Reproducibility** (`system` section):
- `random_seed`: Fixed seed for reproducible results (default: 42)

### Data Assimilation

Configure measurement operators in the `measurement` section:

**Super-resolution:**
```yaml
measurement:
  operator:
    name: super_resolution
    scale_factor: 4  # Downsample by 4x
  noise:
    name: gaussian
    sigma: 0.05
```

**Sparse observations:**
```yaml
measurement:
  operator:
    name: sparse_observation
    ratio: 0.05  # Observe 5% of pixels
  noise:
    name: gaussian
    sigma: 0.05
```

## Command-Line Overrides

Override any config parameter via command line:

```bash
python main.py --config configs/config_training.yaml \
    --time_window 15 \
    --auto_step 5 \
    --grad_scale 2.0 \
    --random_seed 123
```

## Data Format

Expected data format: `[N_trajectories, T, H, W]`
- `N_trajectories`: Number of simulation trajectories
- `T`: Time steps per trajectory
- `H, W`: Spatial dimensions (e.g., 128×128)

Load your data in `configs/config_training.yaml`:
```yaml
dataset:
  nse:
    nse_datapath: '/path/to/your/data.pt'
```

## Outputs

### During Training
- **Checkpoints**: Saved to `ckpts_nse_win=<time_window>_<timestamp>/`
- **Samples**: Visualization images saved to `tmp_images/`
- **WandB Logging**: Training metrics and visualizations (if enabled)

### During Inference
- **Predictions**: Red-blue colormapped images showing:
  - Row 0: Conditioning frames (z0)
  - Row 1: Noisy measurements (y)
  - Row 2: Model predictions (sample)
  - Row 3: Ground truth (z1)
- **Layout**: Each column = one time step, each row = different type

## Examples

### Train with custom settings

```bash
python main.py --config configs/config_training.yaml \
    --time_window 20 \
    --batch_size 16 \
    --base_lr 1e-5
```

### Inference on specific scenario

```bash
python main.py --config configs/config_inference.yaml \
    --trajectory_index 0 \
    --time_index 0 \
    --auto_step 10 \
    --EM_sample_steps 500
```


