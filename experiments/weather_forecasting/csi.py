import torch
import numpy as np
import os
import glob

# --- Configuration - Adjust these as needed ---
FOLDER_PATH = "/scratch/qingqu_root/qingqu/jiayx/forecasting/navier-stokes/sevir_0522_fno/"  # <--- USER: SPECIFY THE CORRECT PATH TO YOUR DATA
X_PRIME_PATTERN = "ans_expid4_repeat*_step0.npy" # These are your "already optimized" x
GROUND_TRUTH_Z_FILE_BASENAME = "gt_expid4_repeat1_step0.npy" # The specific Z file

OUTPUT_DIR_NAME = "direct_evaluation_metrics" # Directory for saving results

def calculate_csi(predictions, ground_truth, threshold):
    """
    Calculates the Critical Success Index (CSI) for each item in a batch.

    Args:
        predictions (torch.Tensor): The predicted/optimized values. Shape: [B, C, H, W]
        ground_truth (torch.Tensor): The ground truth values. Shape: [B, C, H, W] (or [1, C, H, W] for broadcasting)
        threshold (float): The threshold for binary classification.

    Returns:
        torch.Tensor: A tensor of CSI values, one for each item in the batch. Shape: [B]
    """
    # Ensure ground_truth is broadcastable to predictions shape if necessary
    if ground_truth.shape[0] == 1 and predictions.shape[0] > 1:
        ground_truth = ground_truth.expand_as(predictions)
    elif ground_truth.shape != predictions.shape:
         # This condition handles cases where ground_truth is not [1,C,H,W] and also not already [B,C,H,W] matching predictions
        if predictions.shape[0] == 1 and ground_truth.shape[0] == 1 : # Both are single items, check spatial/channel
            if predictions.shape[1:] != ground_truth.shape[1:]:
                raise ValueError(f"Shape mismatch for CSI (spatial/channel dims): predictions {predictions.shape}, ground_truth {ground_truth.shape}")
        # If ground_truth.shape[0] != 1 and shapes still don't match (e.g. different batch sizes)
        elif ground_truth.shape[0] != 1 :
             raise ValueError(f"Shape mismatch for CSI: predictions {predictions.shape}, ground_truth {ground_truth.shape}")
        # If all checks pass, shapes are compatible or ground_truth was successfully expanded.

    # Binary masks based on the threshold
    pred_over_thresh = (predictions >= threshold).float()
    gt_over_thresh = (ground_truth >= threshold).float()

    # Sum over C, H, W dimensions (all dimensions except batch dim 0)
    # to get per-batch-item counts.
    # tuple(range(1, predictions.ndim)) will be (1,2,3) for 4D tensors [B,C,H,W]
    sum_dims = tuple(range(1, predictions.ndim))

    hits = torch.sum(pred_over_thresh * gt_over_thresh, dim=sum_dims)
    misses = torch.sum((1 - pred_over_thresh) * gt_over_thresh, dim=sum_dims)
    false_alarms = torch.sum(pred_over_thresh * (1 - gt_over_thresh), dim=sum_dims)

    # CSI calculation per item
    denominator = hits + misses + false_alarms
    
    csi_per_item = torch.zeros_like(denominator, dtype=torch.float32) # Ensure float for division
    
    # Avoid division by zero: only calculate CSI where denominator is positive
    valid_denominator_mask = denominator > 0
    
    csi_per_item[valid_denominator_mask] = hits[valid_denominator_mask] / denominator[valid_denominator_mask]
    # For items where denominator is 0, CSI remains 0, which is a standard way to handle it.
    
    return csi_per_item

def main_evaluate_directly():
    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create output directory ---
    output_path = os.path.join(FOLDER_PATH, OUTPUT_DIR_NAME)
    os.makedirs(output_path, exist_ok=True)
    print(f"Evaluation results will be saved in: {output_path}")

    # --- 1. Load all x_prime files (considered as "optimized_x") and stack them ---
    x_prime_files = sorted(glob.glob(os.path.join(FOLDER_PATH, X_PRIME_PATTERN)))
    if not x_prime_files:
        print(f"Error: No x_prime files found matching pattern '{X_PRIME_PATTERN}' in '{FOLDER_PATH}'.")
        print("Please check FOLDER_PATH and X_PRIME_PATTERN.")
        return

    print(f"Found {len(x_prime_files)} x_prime files (these will be evaluated directly).")
    x_evaluated_list = []
    for f_path in x_prime_files:
        try:
            x_p_loaded = np.load(f_path)
            if x_p_loaded.ndim == 4 and x_p_loaded.shape[0] == 1: # Handles [1, C, H, W]
                   x_p = x_p_loaded[0]
            elif x_p_loaded.ndim == 3: # Assumes [C, H, W]
                   x_p = x_p_loaded
            else:
                # Fallback to original behavior if shape is unexpected, might need adjustment
                print(f"Warning: Unexpected shape {x_p_loaded.shape} for {f_path}. Using loaded_array[0].")
                x_p = x_p_loaded[0]
            x_evaluated_list.append(torch.from_numpy(x_p).float())
        except Exception as e:
            print(f"Could not load or process {f_path}: {e}")
            return

    if not x_evaluated_list:
        print("Error: No x_prime files were successfully loaded for evaluation.")
        return

    try:
        # This is your batch of "already optimized" data
        evaluated_x_batch_tensor = torch.stack(x_evaluated_list, dim=0).to(device)
    except RuntimeError as e:
        print(f"Error stacking x_prime (evaluated_x) tensors: {e}")
        print("This usually happens if the individual .npy files do not have the exact same shape after loading.")
        for i, x_p_tensor in enumerate(x_evaluated_list):
            print(f"   Shape of loaded data from {x_prime_files[i]}: {x_p_tensor.shape}")
        return

    print(f"evaluated_x_batch_tensor shape: {evaluated_x_batch_tensor.shape}") # Expected: [B, C, H, W]

    # --- 2. Load Ground Truth (z) ---
    print('scale check',evaluated_x_batch_tensor.min(),evaluated_x_batch_tensor.max())
    gt_z_filepath = os.path.join(FOLDER_PATH, GROUND_TRUTH_Z_FILE_BASENAME)
    if not os.path.exists(gt_z_filepath):
        print(f"Error: Ground truth Z file not found: {gt_z_filepath}")
        return
    try:
        z_gt_np = np.load(gt_z_filepath)
        if z_gt_np.ndim == 4 and z_gt_np.shape[0] == 1: # Already [1, C, H, W]
            pass
        elif z_gt_np.ndim == 3: # [C, H, W]
            z_gt_np = np.expand_dims(z_gt_np, axis=0)
        elif z_gt_np.ndim == 2: # [H,W], assume C=1
             z_gt_np = np.expand_dims(np.expand_dims(z_gt_np, axis=0), axis=0) # to [1,1,H,W]
        else: # Attempt to handle common cases or use slice
            print(f"Warning: Ground truth z shape {z_gt_np.shape} is not standard [1,C,H,W] or [C,H,W]. Attempting to use z_gt_np[:1].")
            z_gt_np = z_gt_np[:1] # Take the first slice, hoping it's [1,C,H,W] or similar

        z_gt_tensor = torch.from_numpy(z_gt_np).float().to(device)

        # Ensure z_gt_tensor is [1, C, H, W] for broadcasting against the batch
        if z_gt_tensor.shape[0] != 1:
            print(f"Warning: z_gt_tensor first dimension is not 1 after loading, taking the first slice. Original shape: {z_gt_tensor.shape}")
            z_gt_tensor = z_gt_tensor[0:1]
        if z_gt_tensor.ndim != 4:
             raise ValueError(f"Ground truth tensor must be 4D [1, C, H, W] for broadcasting, but got {z_gt_tensor.shape} after processing.")
        # Ensure spatial and channel dimensions match the evaluated_x batch
        if z_gt_tensor.shape[1:] != evaluated_x_batch_tensor.shape[1:]:
            raise ValueError(f"Channel/Spatial dimensions mismatch between ground truth {z_gt_tensor.shape[1:]} and x_primes {evaluated_x_batch_tensor.shape[1:]}")


    except Exception as e:
        print(f"Could not load or process ground truth Z file {gt_z_filepath}: {e}")
        return

    print(f"Ground truth z_gt_tensor shape: {z_gt_tensor.shape}")

    # --- 3. Calculate Metrics directly ---
    # Difference: evaluated_x_batch_tensor (our "optimized" inputs) vs z_gt_tensor
    difference_from_gt_batch = evaluated_x_batch_tensor - z_gt_tensor # Broadcasting [B,C,H,W] - [1,C,H,W]
    print(f"Difference (evaluated_x - z_gt) batch shape: {difference_from_gt_batch.shape}")

    # --- Summary Statistics ---
    sum_of_differences = torch.sum(difference_from_gt_batch).item()
    sum_of_absolute_differences = torch.sum(torch.abs(difference_from_gt_batch)).item()

    # Overall MAE (mean over all pixels in the batch)
    mean_absolute_error_overall = torch.mean(torch.abs(difference_from_gt_batch)).item()

    # Overall RMSE (root of mean of all squared errors in the batch)
    squared_errors_all_pixels = torch.square(difference_from_gt_batch)
    mean_squared_error_overall = torch.mean(squared_errors_all_pixels).item()
    rmse_overall = np.sqrt(mean_squared_error_overall)

    # --- Calculate RMSE per item in the batch and then its Mean and STD ---
    squared_errors_per_item_pixels = torch.square(difference_from_gt_batch) # Shape [B, C, H, W]
    mse_per_item = torch.mean(squared_errors_per_item_pixels, dim=(1, 2, 3)) # Shape [B]
    rmse_per_item = torch.sqrt(mse_per_item) # Shape [B]

    mean_rmse_across_batch = torch.mean(rmse_per_item).item()
    std_rmse_across_batch = torch.std(rmse_per_item).item() # Std Dev of the per-item RMSEs

    # --- Calculate CSI at specified thresholds ---
    csi_03_per_item = calculate_csi(evaluated_x_batch_tensor, z_gt_tensor, threshold=0.3)
    csi_05_per_item = calculate_csi(evaluated_x_batch_tensor, z_gt_tensor, threshold=0.5)
    mean_csi_03 = torch.mean(csi_03_per_item).item()
    std_csi_03 = torch.std(csi_03_per_item).item() 
    mean_csi_05 = torch.mean(csi_05_per_item).item()
    std_csi_05 = torch.std(csi_05_per_item).item() 

    print("\n--- Direct Evaluation Summary Statistics (No Optimization Step) ---")
    print(f"Data Source (X'): {X_PRIME_PATTERN}")
    print(f"Ground Truth (Z): {GROUND_TRUTH_Z_FILE_BASENAME}")
    print(f"Number of X' samples evaluated: {evaluated_x_batch_tensor.shape[0]}")
    print("--------------------------------------------------------------------")
    print(f"Sum of (X' - GT Z) elements: {sum_of_differences:.4f}")
    print(f"Sum of ABSOLUTE (X' - GT Z) elements: {sum_of_absolute_differences:.4f}")
    print(f"Overall Mean Absolute Error (MAE) (across all pixels in batch): {mean_absolute_error_overall:.6f}")
    print(f"Overall Root Mean Squared Error (RMSE) (across all pixels in batch): {rmse_overall:.6f}")
    print(f"Mean of per-item RMSEs (X' vs GT Z): {mean_rmse_across_batch:.6f}")
    print(f"Std Dev of per-item RMSEs (X' vs GT Z): {std_rmse_across_batch:.6f}")
    print(f"Critical Success Index (CSI) at threshold 0.3: {mean_csi_03:.6f}")
    print(f"Std Dev of CSI at threshold 0.3: {std_csi_03:.6f}")
    print(f"Critical Success Index (CSI) at threshold 0.5: {mean_csi_05:.6f}")
    print(f"Std Dev of CSI at threshold 0.5: {std_csi_05:.6f}")
    print("--------------------------------------------------------------------\n")

    # --- Save results to a file ---
    results_summary = {
        "x_prime_pattern": X_PRIME_PATTERN,
        "ground_truth_file": GROUND_TRUTH_Z_FILE_BASENAME,
        "num_samples": evaluated_x_batch_tensor.shape[0],
        "sum_diff": sum_of_differences,
        "sum_abs_diff": sum_of_absolute_differences,
        "mae_overall": mean_absolute_error_overall,
        "rmse_overall": rmse_overall,
        "mean_rmse_per_item": mean_rmse_across_batch,
        "std_rmse_per_item": std_rmse_across_batch,
        "csi_0.3": csi_03,
        "csi_0.5": csi_05,
        "csi_0.7": csi_07,
    }

    results_file_path = os.path.join(output_path, "direct_evaluation_summary.txt")
    with open(results_file_path, "w") as f:
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Direct evaluation summary saved to {results_file_path}")

    # --- Optional: Save difference images if needed ---
    # This would save the (x_prime - z_gt) for each x_prime.
    # For example, save the first few difference images:
    # num_diff_to_save = min(5, evaluated_x_batch_tensor.shape[0])
    # for i in range(num_diff_to_save):
    #     orig_filename_base = os.path.splitext(os.path.basename(x_prime_files[i]))[0]
    #     diff_image_np = difference_from_gt_batch[i].cpu().numpy()
    #     np.save(os.path.join(output_path, f"diff_gt_{orig_filename_base}.npy"), diff_image_np)
    # print(f"Saved first {num_diff_to_save} difference images to {output_path}")


if __name__ == "__main__":
    if FOLDER_PATH == "/scratch/qingqu_root/qingqu1/siyiche/FNO_multibreast/results/sevir_fno_numerical_0520/" and not os.path.exists(FOLDER_PATH):
           print(f"WARNING: The default FOLDER_PATH '{FOLDER_PATH}' does not exist.")
           print("Please update the 'FOLDER_PATH' variable in the script to your actual data directory if this is not a test run where the path is mocked/unused.")

    if FOLDER_PATH == "path/to/your/data_folder": # Default check
        print("ERROR: Please update the 'FOLDER_PATH' variable in the script to your actual data directory.")
    else:
        main_evaluate_directly()