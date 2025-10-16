#!/usr/bin/env python
"""
Simple test script to verify the configuration system works correctly.
Run this to check if config loading and overrides work as expected.
"""

import yaml
from utils import Config

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def test_config_loading():
    """Test basic config loading from YAML"""
    print("=" * 70)
    print("TEST 1: Loading config from YAML file")
    print("=" * 70)
    
    config_dict = load_yaml('./configs/config.yaml')
    conf = Config(config_dict=config_dict, device='cpu')
    
    print(f"✓ Dataset: {conf.dataset}")
    print(f"✓ Batch size: {conf.batch_size}")
    print(f"✓ Time window: {conf.time_window}")
    print(f"✓ UNet channels: {conf.unet_channels}")
    print(f"✓ Max steps: {conf.max_steps}")
    print(f"✓ Sigma coef: {conf.sigma_coef}")
    print(f"✓ Beta function: {conf.beta_fn}")
    
    assert conf.dataset == 'nse', "Dataset should be 'nse'"
    assert conf.batch_size == 32, "Default batch size should be 32 for NSE"
    assert conf.time_window == 1, "Default time window should be 1"
    
    print("\n✅ Test 1 PASSED: Config loaded successfully from YAML\n")

def test_command_line_overrides():
    """Test that command-line arguments override config file values"""
    print("=" * 70)
    print("TEST 2: Command-line argument overrides")
    print("=" * 70)
    
    config_dict = load_yaml('./configs/config.yaml')
    
    # Simulate command-line overrides
    conf = Config(
        config_dict=config_dict,
        device='cpu',
        time_window=10,  # Override from default 1
        sigma_coef=2.0,  # Override from default 1.0
        batch_size=64    # This won't override (not in kwargs handling)
    )
    
    print(f"✓ Time window (overridden): {conf.time_window}")
    print(f"✓ Sigma coef (overridden): {conf.sigma_coef}")
    
    assert conf.time_window == 10, "Time window should be overridden to 10"
    assert conf.sigma_coef == 2.0, "Sigma coef should be overridden to 2.0"
    
    print("\n✅ Test 2 PASSED: Command-line overrides work correctly\n")

def test_dataset_specific_config():
    """Test that dataset-specific configs are loaded correctly"""
    print("=" * 70)
    print("TEST 3: Dataset-specific configuration")
    print("=" * 70)
    
    config_dict = load_yaml('./configs/config.yaml')
    
    # Test NSE config
    conf_nse = Config(config_dict=config_dict, dataset='nse', device='cpu')
    print(f"✓ NSE - Hi size: {conf_nse.hi_size}")
    print(f"✓ NSE - Lo size: {conf_nse.lo_size}")
    print(f"✓ NSE - Time lag: {conf_nse.time_lag}")
    print(f"✓ NSE - Center data: {conf_nse.center_data}")
    
    assert conf_nse.hi_size == 128, "NSE hi_size should be 128"
    assert conf_nse.center_data == False, "NSE should not center data"
    
    print("\n✅ Test 3 PASSED: Dataset-specific config works correctly\n")

def test_debug_mode():
    """Test that debug mode overrides work"""
    print("=" * 70)
    print("TEST 4: Debug mode configuration")
    print("=" * 70)
    
    config_dict = load_yaml('./configs/config.yaml')
    
    # Normal mode
    conf_normal = Config(config_dict=config_dict, device='cpu', debug=False)
    print(f"✓ Normal - EM steps: {conf_normal.EM_sample_steps}")
    print(f"✓ Normal - Sample every: {conf_normal.sample_every}")
    
    # Debug mode
    conf_debug = Config(config_dict=config_dict, device='cpu', debug=True)
    print(f"✓ Debug - EM steps: {conf_debug.EM_sample_steps}")
    print(f"✓ Debug - Sample every: {conf_debug.sample_every}")
    
    assert conf_normal.EM_sample_steps == 500, "Normal mode should use 500 steps"
    assert conf_debug.EM_sample_steps == 10, "Debug mode should use 10 steps"
    assert conf_debug.sample_every == 10, "Debug mode should sample every 10 steps"
    
    print("\n✅ Test 4 PASSED: Debug mode overrides work correctly\n")

def test_model_architecture():
    """Test that model architecture config is loaded"""
    print("=" * 70)
    print("TEST 5: Model architecture configuration")
    print("=" * 70)
    
    config_dict = load_yaml('./configs/config.yaml')
    conf = Config(config_dict=config_dict, device='cpu')
    
    print(f"✓ UNet channels: {conf.unet_channels}")
    print(f"✓ UNet dim_mults: {conf.unet_dim_mults}")
    print(f"✓ ResNet groups: {conf.unet_resnet_block_groups}")
    print(f"✓ Attention heads: {conf.unet_attn_heads}")
    print(f"✓ Attention dim head: {conf.unet_attn_dim_head}")
    
    assert conf.unet_channels == 128, "UNet channels should be 128"
    assert conf.unet_dim_mults == (1, 2, 2, 2), "UNet dim_mults should be (1,2,2,2)"
    assert conf.unet_attn_heads == 4, "Attention heads should be 4"
    
    print("\n✅ Test 5 PASSED: Model architecture config loaded correctly\n")

def test_missing_config_defaults():
    """Test that missing config values use defaults"""
    print("=" * 70)
    print("TEST 6: Default values for missing config entries")
    print("=" * 70)
    
    # Empty config dict
    empty_config = {}
    conf = Config(config_dict=empty_config, device='cpu')
    
    print(f"✓ Dataset (default): {conf.dataset}")
    print(f"✓ Sigma coef (default): {conf.sigma_coef}")
    print(f"✓ Beta function (default): {conf.beta_fn}")
    print(f"✓ Max steps (default): {conf.max_steps}")
    
    assert conf.dataset == 'nse', "Default dataset should be 'nse'"
    assert conf.sigma_coef == 1.0, "Default sigma_coef should be 1.0"
    assert conf.beta_fn == 't^2', "Default beta_fn should be 't^2'"
    
    print("\n✅ Test 6 PASSED: Default values work correctly\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("FLOWDAS CONFIGURATION SYSTEM TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_config_loading()
        test_command_line_overrides()
        test_dataset_specific_config()
        test_debug_mode()
        test_model_architecture()
        test_missing_config_defaults()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED! Configuration system is working correctly.")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())

