import torch

def show_torch_device():
    """Display PyTorch device information"""
    
    print("=" * 50)
    print("PyTorch Device Information")
    print("=" * 50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA device count
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Device name: {device_name}")
        
        # Show device capabilities
        device_capability = torch.cuda.get_device_capability(current_device)
        print(f"Device capability: {device_capability}")
        
        # Default device is GPU
        device = torch.device("cuda")
        print(f"\n✓ Using GPU: {device_name}")
        
    else:
        # Default device is CPU
        device = torch.device("cpu")
        print("\n✓ Using CPU")
    
    print("=" * 50)
    
    return device

# Run the function
device = show_torch_device()

# Example tensor operations on the detected device
print("\nExample tensor operations:")
x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x @ y  # Matrix multiplication
print(f"Tensor on: {z.device}")
print(f"Result:\n{z}")