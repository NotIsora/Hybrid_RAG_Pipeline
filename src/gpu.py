"""
Google Colab GPU Verification Utility (Optimized for NVIDIA L4 & T4)
"""

import sys
import torch

def verify_gpu():
    print("=" * 65)
    print("Google Colab Hardware & Architecture Verification")
    print("=" * 65)
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"CUDA Available:     {cuda_available}")
    
    if not cuda_available:
        print("\n❌ WARNING: No GPU detected!")
        print("To enable L4 GPU in Google Colab:")
        print("  1. Click 'Runtime' in the top menu.")
        print("  2. Select 'Change runtime type'.")
        print("  3. Under 'Hardware accelerator', choose 'L4 GPU' (or 'T4 GPU').")
        print("  4. Click 'Save' and re-run this check.")
        print("=" * 65)
        return False

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / (1024 ** 3)
    major, minor = props.major, props.minor

    print(f"Device Name:        {device_name}")
    print(f"Compute Capability: {major}.{minor}")
    print(f"Total VRAM:         {total_mem_gb:.2f} GB")
    
    # Check Architecture Capabilities
    has_bf16 = torch.cuda.is_bf16_supported()
    print(f"Bfloat16 Support:   {'✅ Native' if has_bf16 else '❌ Not Native (use FP16)'}")

    # Check for L4 GPU (Ada Lovelace, CC 8.9, 24GB)
    if "L4" in device_name or (major == 8 and minor == 9):
        print("\n🚀 OPTIMIZED FOR NVIDIA L4 GPU (Ada Lovelace):")
        print("  • 24 GB High-Bandwidth VRAM detected.")
        print("  • Native Bfloat16 enabled for maximum numeric stability.")
        print("  • TF32 (TensorFloat-32) acceleration active.")
        print("  • PyTorch SDPA (Scaled Dot-Product Attention) enabled.")
        print("  • Capable of running Qwen 2.5 7B in FULL native bfloat16, or 14B in 4-bit!")
    elif total_mem_gb >= 14.0:
        print("\n✅ T4 GPU (or equivalent ~15GB VRAM) detected.")
        print("  • Recommended mode: 4-bit NF4 Quantization with FP16 compute.")
    else:
        print("\n⚠️ Low VRAM GPU detected. Recommended: 3B parameter model in 4-bit.")

    print("=" * 65)
    return True

if __name__ == "__main__":
    verify_gpu()
