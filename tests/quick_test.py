#!/usr/bin/env python3
"""
Quick test to verify authentication and basic functionality
without downloading the full model
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

def test_dependencies():
    """Test that all required dependencies are available"""
    print("Testing dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not installed")
        return False
        
    try:
        import accelerate
        print(f"✅ Accelerate: {accelerate.__version__}")
    except ImportError:
        print("❌ Accelerate not installed")
        return False
    
    return True

def test_authentication():
    """Test Hugging Face authentication"""
    print("\nTesting authentication...")
    
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
    
    if not token:
        print("❌ No Hugging Face token found")
        return False
    
    print(f"✅ Found token: {token[:10]}...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.model_info("microsoft/maira-2")
        print("✅ Authentication successful")
        print(f"✅ Access to MAIRA-2 confirmed")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def test_image_processing():
    """Test basic image processing functionality"""
    print("\nTesting image processing...")
    
    try:
        # Create a sample grayscale chest X-ray-like image
        image_array = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        # Add some structure that mimics a chest X-ray
        # Ribcage-like pattern
        for i in range(5):
            y = 100 + i * 50
            image_array[y:y+5, :] = 200
        
        # Lung field areas (darker)
        image_array[150:400, 100:200] = np.clip(image_array[150:400, 100:200] * 0.7, 0, 255).astype(np.uint8)
        image_array[150:400, 300:400] = np.clip(image_array[150:400, 300:400] * 0.7, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        test_image = Image.fromarray(image_array, mode='L')
        
        # Test image normalization (like in MAIRA-2 processor)
        normalized_array = np.array(test_image.convert("L"))
        normalized_array = normalized_array.astype(float)
        normalized_array -= normalized_array.min()
        if normalized_array.max() > 0:
            normalized_array /= normalized_array.max()
        normalized_array *= 255
        normalized_array = normalized_array.astype(np.uint8)
        
        normalized_image = Image.fromarray(normalized_array).convert("L")
        
        print("✅ Image processing test passed")
        print(f"   Original size: {test_image.size}")
        print(f"   Normalized range: {normalized_array.min()}-{normalized_array.max()}")
        
        return True, test_image, normalized_image
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False, None, None

def test_visualization_setup():
    """Test matplotlib visualization setup"""
    print("\nTesting visualization setup...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Create a simple test plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Test Attention Visualization Layout')
        
        # Simulate attention data
        attention_data = np.random.rand(50)
        image_attention_2d = np.random.rand(24, 24)
        text_attention = np.random.rand(20)
        
        # Plot 1: Full attention
        axes[0, 0].plot(attention_data)
        axes[0, 0].set_title('Full Attention Pattern')
        axes[0, 0].set_xlabel('Token Position')
        axes[0, 0].set_ylabel('Attention Weight')
        
        # Plot 2: Image attention
        im = axes[0, 1].imshow(image_attention_2d, cmap='Reds')
        axes[0, 1].set_title('Image Attention (24x24)')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: Text attention
        axes[1, 0].bar(range(len(text_attention)), text_attention)
        axes[1, 0].set_title('Text Token Attention')
        axes[1, 0].set_xlabel('Text Token Position')
        axes[1, 0].set_ylabel('Attention Weight')
        
        # Plot 4: Stats
        stats_text = """
        Test Statistics:
        Max: 0.987
        Min: 0.023
        Mean: 0.456
        Std: 0.234
        
        Test Structure:
        Images: 1
        Tokens: 50
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('Attention Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save test visualization in tests folder
        test_dir = Path(__file__).parent
        plt.savefig(test_dir / 'test_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization test passed")
        print("✅ Test plot saved as 'tests/test_visualization.png'")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MAIRA-2 Attention Visualizer - Quick Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Dependencies
    if not test_dependencies():
        success = False
    
    # Test 2: Authentication
    if not test_authentication():
        success = False
    
    # Test 3: Image processing
    img_success, original, normalized = test_image_processing()
    if not img_success:
        success = False
    else:
        # Save test images in tests folder
        try:
            test_dir = Path(__file__).parent
            original.save(test_dir / 'test_original.png')
            normalized.save(test_dir / 'test_normalized.png')
            print("✅ Test images saved: tests/test_original.png, tests/test_normalized.png")
        except Exception as e:
            print(f"⚠️  Could not save test images: {e}")
    
    # Test 4: Visualization
    if not test_visualization_setup():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Your environment is ready for MAIRA-2 attention visualization.")
        print("\nNext steps:")
        print("1. Run the full visualizer: uv run python attention_visualizer.py")
        print("2. Or import and use in your code:")
        print("   from attention_visualizer import MAIRA2AttentionVisualizer")
        print("3. Run tests: uv run python tests/quick_test.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()