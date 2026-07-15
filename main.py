import os
import sys
import yaml
import torch

from src.utils.seed import set_seed

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("=" * 60)
    print("AI Security Copilot - Environment Verification")
    print("=" * 60)
    
    # 1. Load config
    try:
        config = load_config()
        print("✓ Configuration file loaded successfully.")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)
        
    # 2. Set Seed for Reproducibility
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)
    
    # 3. Verify Python and package environment
    print("\n[Environment Details]")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check imports of other packages to verify installation
    packages = ["transformers", "sklearn", "pandas", "numpy", "matplotlib", "fastapi", "uvicorn", "pytest"]
    print("\n[Package Import Checks]")
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}: Imported successfully")
        except ImportError:
            print(f"  ✗ {pkg}: Failed to import")
            
    # 4. Check CUDA / GPU availability
    print("\n[Hardware Details]")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Index: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        # CUDA capability
        major, minor = torch.cuda.get_device_capability()
        print(f"CUDA Capability: {major}.{minor}")
    else:
        print("Running on CPU. GPU acceleration is not available.")
        
    print("=" * 60)
    print("Verification completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
