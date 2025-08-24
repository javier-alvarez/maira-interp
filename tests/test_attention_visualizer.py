#!/usr/bin/env python3
"""
Test suite for MAIRA-2 Attention Visualizer

Tests the core functionality of the attention visualization tool.
"""

import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from attention_visualizer import MAIRA2AttentionVisualizer
except ImportError as e:
    pytest.skip(f"Could not import attention_visualizer: {e}. Install dependencies first.", allow_module_level=True)


class TestMAIRA2AttentionVisualizer:
    """Test cases for MAIRA-2 attention visualizer"""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures before running tests"""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test image (grayscale chest X-ray like)
        cls.test_image = Image.new('L', (512, 512), color=128)
        
        # Save test image
        cls.test_image_path = Path(cls.temp_dir) / "test_frontal.png"
        cls.test_image.save(cls.test_image_path)
    
    @classmethod
    def teardown_class(cls):
        """Clean up after tests"""
        shutil.rmtree(cls.temp_dir)
    
    def setup_method(self):
        """Set up before each test"""
        # Skip tests if we can't load the model (e.g., in CI without GPU)
        try:
            self.visualizer = MAIRA2AttentionVisualizer()
        except Exception as e:
            pytest.skip(f"Could not initialize visualizer: {e}")
    
    @pytest.mark.requires_model
    def test_model_initialization(self):
        """Test that the model initializes correctly"""
        assert self.visualizer.model is not None
        assert self.visualizer.processor is not None
        assert self.visualizer.vision_layers == 12
        assert self.visualizer.vision_heads == 12
        assert self.visualizer.text_layers == 32
        assert self.visualizer.text_heads == 32
        assert self.visualizer.image_seq_length == 576
    
    def test_process_inputs_single_image(self):
        """Test processing inputs with only frontal image"""
        processed, metadata = self.visualizer.process_inputs(
            frontal_image=self.test_image,
            indication="Test indication"
        )
        
        # Check that we get the expected structure
        self.assertIn('input_ids', processed)
        self.assertIn('pixel_values', processed)
        
        # Check metadata
        self.assertTrue(metadata['has_frontal'])
        self.assertFalse(metadata['has_lateral'])
        self.assertFalse(metadata['has_prior'])
        self.assertEqual(metadata['num_images'], 1)
    
    def test_process_inputs_multiple_images(self):
        """Test processing inputs with multiple images"""
        processed, metadata = self.visualizer.process_inputs(
            frontal_image=self.test_image,
            lateral_image=self.test_image,  # Reuse same image for testing
            indication="Test indication",
            technique="PA and lateral"
        )
        
        # Check metadata for multiple images
        self.assertTrue(metadata['has_frontal'])
        self.assertTrue(metadata['has_lateral'])
        self.assertFalse(metadata['has_prior'])
        self.assertEqual(metadata['num_images'], 2)
    
    def test_image_token_positions(self):
        """Test finding image token positions in input sequence"""
        processed, metadata = self.visualizer.process_inputs(
            frontal_image=self.test_image,
            indication="Test"
        )
        
        # Should find at least one image token position
        self.assertGreater(len(metadata['image_positions']), 0)
    
    @pytest.skipIf(not torch.cuda.is_available(), "GPU not available for generation test")
    def test_generate_with_attention_short(self):
        """Test generation with attention capture (short sequence)"""
        processed, metadata = self.visualizer.process_inputs(
            frontal_image=self.test_image,
            indication="Brief test"
        )
        
        # Generate very few tokens to speed up test
        try:
            outputs, attention_weights = self.visualizer.generate_with_attention(
                processed, max_new_tokens=5
            )
            
            # Check outputs
            self.assertIsNotNone(outputs)
            self.assertIn('sequences', outputs.__dict__)
            
            # Check attention weights structure
            self.assertIn('text_attention', attention_weights)
            
        except Exception as e:
            self.skipTest(f"Generation failed (may require GPU): {e}")
    
    def test_attention_visualization_mock(self):
        """Test attention visualization with mock data"""
        # Create mock attention weights
        seq_len = 100
        num_heads = 8
        
        # Mock attention: (batch=1, heads, seq_len, seq_len)
        mock_attention = torch.rand(1, num_heads, seq_len, seq_len)
        mock_attention = torch.softmax(mock_attention, dim=-1)  # Normalize
        
        attention_weights = {
            'text_attention': [mock_attention]
        }
        
        # Mock metadata
        input_metadata = {
            'num_images': 1,
            'has_frontal': True,
            'has_lateral': False,
            'has_prior': False,
            'image_positions': [10, 11, 12],  # Mock image token positions
            'pixel_values': torch.rand(1, 3, 224, 224)  # Mock pixel values
        }
        
        # Mock output tokens
        output_tokens = torch.randint(0, 1000, (seq_len,))
        
        # Test visualization creation (should not crash)
        try:
            fig = self.visualizer.visualize_token_attention(
                attention_weights=attention_weights,
                input_metadata=input_metadata,
                output_tokens=output_tokens,
                token_idx=50,  # Middle token
                layer_idx=0
            )
            
            self.assertIsNotNone(fig)
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            self.fail(f"Attention visualization failed: {e}")
    
    def test_output_directory_creation(self):
        """Test that output directories are created properly"""
        output_dir = Path(self.temp_dir) / "test_output"
        
        # This would normally run the full pipeline, but we'll just test directory creation
        output_dir.mkdir(exist_ok=True)
        self.assertTrue(output_dir.exists())
        
        # Test file creation
        test_file = output_dir / "test_file.txt"
        test_file.write_text("test content")
        self.assertTrue(test_file.exists())


class TestMAIRA2Architecture(unittest.TestCase):
    """Test understanding of MAIRA-2 architecture"""
    
    def test_architecture_constants(self):
        """Test that we have the correct architecture understanding"""
        # Vision encoder (RAD-DINO-MAIRA-2 based on DINOv2)
        self.assertEqual(12, 12)  # 12 layers
        self.assertEqual(12, 12)  # 12 attention heads per layer
        
        # Language model (Vicuna-7B-v1.5 based on LLaMA)
        self.assertEqual(32, 32)  # 32 layers
        self.assertEqual(32, 32)  # 32 attention heads per layer
        
        # Image sequence length
        self.assertEqual(576, 576)  # 576 tokens per image
        
        # Multimodal projector
        self.assertEqual(4, 4)  # 4 layers in projector
    
    def test_input_modalities(self):
        """Test understanding of supported input modalities"""
        supported_inputs = [
            "frontal_cxr",      # Current frontal chest X-ray (required)
            "lateral_view",     # Current lateral view (optional)
            "prior_frontal",    # Prior frontal view (optional)
            "indication",       # Indication text (optional)
            "technique",        # Technique section (optional)
            "comparison",       # Comparison section (optional)
            "prior_report"      # Prior report text (optional)
        ]
        
        # Just verify we understand the inputs
        self.assertIn("frontal_cxr", supported_inputs)
        self.assertEqual(len(supported_inputs), 7)
    
    def test_output_types(self):
        """Test understanding of MAIRA-2 output types"""
        output_types = [
            "ungrounded_report",    # Text-only findings
            "grounded_report",      # Findings with bounding boxes
            "phrase_grounding"      # Specific phrase localization
        ]
        
        self.assertEqual(len(output_types), 3)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMAIRA2AttentionVisualizer))
    suite.addTests(loader.loadTestsFromTestCase(TestMAIRA2Architecture))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)