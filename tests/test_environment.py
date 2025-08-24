#!/usr/bin/env python3
"""
Environment and dependency tests for MAIRA-2 Attention Visualizer

These tests verify that the environment is properly set up and all
dependencies are available. They should work in VS Code pytest plugin.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEnvironment:
    """Test that the environment is properly configured"""
    
    def test_python_version(self):
        """Test that Python version is adequate"""
        assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"
    
    def test_dependencies_importable(self):
        """Test that core dependencies can be imported"""
        
        # Test PyTorch
        try:
            import torch
            assert hasattr(torch, '__version__')
        except ImportError:
            pytest.fail("PyTorch not available")
        
        # Test Transformers
        try:
            import transformers
            assert hasattr(transformers, '__version__')
            # Check minimum version
            from packaging import version
            assert version.parse(transformers.__version__) >= version.parse("4.48.0")
        except ImportError:
            pytest.fail("Transformers not available")
        
        # Test PIL
        try:
            from PIL import Image
            assert hasattr(Image, 'new')
        except ImportError:
            pytest.fail("PIL/Pillow not available")
        
        # Test matplotlib
        try:
            import matplotlib.pyplot as plt
            assert hasattr(plt, 'subplots')
        except ImportError:
            pytest.fail("Matplotlib not available")
        
        # Test numpy
        try:
            import numpy as np
            assert hasattr(np, 'array')
        except ImportError:
            pytest.fail("NumPy not available")
    
    def test_huggingface_auth(self):
        """Test that Hugging Face authentication is available"""
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv(project_root / ".env")
        
        # Check for token
        token = (os.getenv("HF_TOKEN") or 
                os.getenv("HUGGING_FACE_HUB_TOKEN") or 
                os.getenv("HF_HUB_TOKEN"))
        
        if not token:
            pytest.skip("No Hugging Face token found. Set HF_TOKEN environment variable.")
        
        assert len(token) > 10, "Token seems too short"
        assert token.startswith("hf_"), "Token should start with 'hf_'"
    
    def test_project_structure(self):
        """Test that project structure is correct"""
        
        # Check main files exist
        assert (project_root / "attention_visualizer.py").exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "README.md").exists()
        
        # Check tests directory
        tests_dir = project_root / "tests"
        assert tests_dir.exists()
        assert (tests_dir / "__init__.py").exists()
        
        # Check for .env file (should exist for this project)
        env_file = project_root / ".env"
        if not env_file.exists():
            pytest.skip("No .env file found. Create one with HF_TOKEN for full testing.")
    
    def test_imports_from_main_module(self):
        """Test that we can import from the main module"""
        
        try:
            from attention_visualizer import MAIRA2AttentionVisualizer
            assert MAIRA2AttentionVisualizer is not None
        except ImportError as e:
            # This is expected if dependencies aren't fully installed
            pytest.skip(f"Could not import main module: {e}")


class TestSystemResources:
    """Test system resources and capabilities"""
    
    def test_memory_available(self):
        """Check available system memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Just report, don't fail
            print(f"\nSystem memory: {memory.total / (1024**3):.1f} GB total, {available_gb:.1f} GB available")
            
            if available_gb < 1.0:
                pytest.skip(f"Low memory: {available_gb:.1f} GB available. MAIRA-2 needs 8GB+")
        except ImportError:
            pytest.skip("psutil not available for memory check")
    
    def test_cuda_availability(self):
        """Check CUDA availability"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                print(f"\nCUDA available: {device_count} devices")
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    print(f"  Device {i}: {props.name}, {memory_gb:.1f} GB")
            else:
                print("\nCUDA not available - will use CPU (slower)")
                
        except Exception as e:
            pytest.skip(f"Could not check CUDA: {e}")


@pytest.mark.unit
class TestQuickFunctionality:
    """Quick tests that don't require the full model"""
    
    def test_image_creation(self):
        """Test creating a simple test image"""
        from PIL import Image
        import numpy as np
        
        # Create test image
        image_array = np.ones((512, 512), dtype=np.uint8) * 128
        test_image = Image.fromarray(image_array, mode='L')
        
        assert test_image.size == (512, 512)
        assert test_image.mode == 'L'
    
    def test_visualization_setup(self):
        """Test matplotlib setup"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Add some test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)  # Clean up
    
    def test_output_directory_creation(self):
        """Test creating output directories"""
        import tempfile
        import shutil
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create output structure
            output_dir = temp_dir / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            assert output_dir.exists()
            assert output_dir.is_dir()
            
            # Create a test file
            test_file = output_dir / "test.txt"
            test_file.write_text("test content")
            
            assert test_file.exists()
            assert test_file.read_text() == "test content"
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])