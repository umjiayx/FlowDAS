### 3. Weather Forecasting

To reproduce the results based on our provided code and checkpoint:

```
python main_research_sevir_gen_sample4GL_621_opensource_0710.py \
  --dataset sevir \
  --beta_fn t^2 \
  --sigma_coef 1 \
  --use_wandb 0 \
  --debug 0 \
  --overfit 0 \
  --task_config ./configs/super_resolution_config.yaml \
  --sample_only 1 \
  --load_path <path_to_your_checkpoint> \
  --savedir <path_to_your_output_directory> \
  --sevir_datapath <path_to_your_sevir_dataset> \
  --MC_times 25 \
  --exp_times 100 \
  --auto_step 3 \
  --exp_id_times 1
```




## 📆 Coming Soon
- [x] Checkpoint release
- [x] Inference code release
- [ ] Training code release

