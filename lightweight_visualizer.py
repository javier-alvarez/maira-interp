#!/usr/bin/env python3
"""
Lightweight MAIRA-2 Attention Visualizer for resource-constrained environments

This version uses several optimizations to work with limited RAM:
- Model sharding/offloading
- Reduced precision
- Memory-efficient loading
- Progressive inference
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import os
import psutil
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightMAIRA2Visualizer:
    """Memory-efficient MAIRA-2 attention visualizer for limited resources"""
    
    def __init__(self, model_path: str = "microsoft/maira-2", token: Optional[str] = None):
        """Initialize with memory optimizations"""
        
        # Check system resources
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        logger.info(f"Available RAM: {available_gb:.1f} GB")
        
        if available_gb < 1.0:
            raise RuntimeError(f"Insufficient memory: {available_gb:.1f} GB available, need at least 1GB")
        
        # Get authentication token
        auth_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
        
        if not auth_token:
            raise ValueError("No Hugging Face token found. Set HF_TOKEN environment variable.")
        
        logger.info("Loading MAIRA-2 with memory optimizations...")
        
        # Create offload directory
        offload_dir = Path("./model_offload")
        offload_dir.mkdir(exist_ok=True)
        
        try:
            # Strategy 1: Try with maximum offloading
            if available_gb >= 2.0:
                logger.info("Using disk offloading strategy (RAM >= 2GB)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # Use float16 to save memory
                    device_map="auto",
                    offload_folder=str(offload_dir),
                    offload_state_dict=True,
                    low_cpu_mem_usage=True,
                    token=auth_token
                )
            else:
                # Strategy 2: CPU-only with aggressive memory optimization
                logger.info("Using CPU-only with memory optimization (RAM < 2GB)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # Float32 required for CPU
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    token=auth_token
                )
            
            logger.info("✅ Model loaded successfully!")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                token=auth_token
            )
            
            self.model.eval()
            
            # Model info
            self.vision_layers = 12
            self.vision_heads = 12
            self.text_layers = 32
            self.text_heads = 32
            self.image_seq_length = 576
            
            logger.info("MAIRA-2 ready for lightweight attention visualization")
            
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("GPU out of memory. Try closing other applications or use CPU-only mode.")
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(f"System out of memory: {e}")
            raise
    
    def process_inputs_lightweight(
        self,
        frontal_image: Image.Image,
        lateral_image: Optional[Image.Image] = None,
        indication: Optional[str] = "Chest X-ray evaluation",
        max_length: int = 512  # Limit input length to save memory
    ) -> Tuple[Dict, Dict]:
        """Process inputs with memory constraints"""
        
        logger.info("Processing inputs with memory optimization...")
        
        # Use shorter, simpler prompts to reduce memory
        processed = self.processor.format_and_preprocess_reporting_input(
            current_frontal=frontal_image,
            current_lateral=lateral_image,
            prior_frontal=None,  # Skip prior to save memory
            indication=indication,
            technique="PA chest X-ray",
            comparison="None",
            prior_report=None,
            get_grounding=False,  # Disable grounding to save memory
            return_tensors="pt"
        )
        
        # Truncate if too long
        if processed['input_ids'].shape[-1] > max_length:
            logger.warning(f"Truncating input from {processed['input_ids'].shape[-1]} to {max_length} tokens")
            processed['input_ids'] = processed['input_ids'][:, :max_length]
            if 'attention_mask' in processed:
                processed['attention_mask'] = processed['attention_mask'][:, :max_length]
        
        # Move to appropriate device
        device = next(self.model.parameters()).device
        processed = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in processed.items()}
        
        metadata = {
            'has_frontal': True,
            'has_lateral': lateral_image is not None,
            'has_prior': False,
            'input_length': processed['input_ids'].shape[-1]
        }
        
        return processed, metadata
    
    def generate_lightweight(
        self,
        processed_inputs: Dict,
        max_new_tokens: int = 20,  # Generate fewer tokens to save memory
        temperature: float = 0.1
    ) -> Tuple[torch.Tensor, str]:
        """Generate tokens with memory optimization"""
        
        logger.info(f"Generating {max_new_tokens} tokens...")
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **processed_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    use_cache=False,  # Disable cache to save memory
                    output_attentions=False,  # We'll get attention separately
                    return_dict_in_generate=True
                )
                
                # Decode generated text
                prompt_length = processed_inputs['input_ids'].shape[-1]
                generated_tokens = outputs.sequences[0][prompt_length:]
                generated_text = self.processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()
                
                logger.info(f"Generated: {generated_text}")
                return outputs.sequences[0], generated_text
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("Out of memory during generation. Try reducing max_new_tokens or input length.")
                    raise
                raise
    
    def get_simple_attention(
        self,
        processed_inputs: Dict,
        target_tokens: torch.Tensor,
        layer_idx: int = -1
    ) -> Optional[torch.Tensor]:
        """Get attention weights for analysis (simplified version)"""
        
        logger.info("Computing attention weights...")
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    **processed_inputs,
                    output_attentions=True,
                    use_cache=False
                )
                
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # Get attention from specified layer
                    if layer_idx == -1:
                        layer_idx = len(outputs.attentions) - 1
                    
                    attention = outputs.attentions[layer_idx]  # [batch, heads, seq, seq]
                    
                    # Average across heads to save memory
                    attention_avg = attention.mean(dim=1)  # [batch, seq, seq]
                    
                    return attention_avg[0]  # [seq, seq]
                
                return None
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("Out of memory getting attention. Skipping attention visualization.")
                return None
            raise
    
    def create_simple_visualization(
        self,
        attention: torch.Tensor,
        generated_text: str,
        input_length: int,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create a simple attention visualization"""
        
        if attention is None:
            logger.warning("No attention data available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'MAIRA-2 Attention Visualization\nGenerated: "{generated_text}"')
        
        # Convert to numpy
        attention_np = attention.cpu().numpy()
        
        # Plot 1: Attention heatmap for last few tokens
        ax1 = axes[0]
        # Show attention for last 5 generated tokens to input
        start_idx = max(0, attention_np.shape[0] - 10)
        attention_subset = attention_np[start_idx:, :input_length]
        
        im1 = ax1.imshow(attention_subset, cmap='Blues', aspect='auto')
        ax1.set_title('Attention: Generated Tokens → Input')
        ax1.set_xlabel('Input Position')
        ax1.set_ylabel('Generated Token Position')
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Average attention to input positions
        ax2 = axes[1]
        avg_attention = attention_np[input_length:, :input_length].mean(axis=0)
        ax2.bar(range(len(avg_attention)), avg_attention)
        ax2.set_title('Average Attention to Input Positions')
        ax2.set_xlabel('Input Position')
        ax2.set_ylabel('Average Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def quick_demo(
        self,
        frontal_image: Image.Image,
        lateral_image: Optional[Image.Image] = None,
        output_dir: str = "lightweight_output"
    ):
        """Run a quick demo with minimal memory usage"""
        
        logger.info("Running lightweight MAIRA-2 demo...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Process inputs
            processed, metadata = self.process_inputs_lightweight(
                frontal_image=frontal_image,
                lateral_image=lateral_image,
                indication="Chest X-ray evaluation"
            )
            
            # Generate text
            full_tokens, generated_text = self.generate_lightweight(
                processed, max_new_tokens=15
            )
            
            # Get attention
            attention = self.get_simple_attention(
                processed, full_tokens, layer_idx=-1
            )
            
            # Create visualization
            fig = self.create_simple_visualization(
                attention=attention,
                generated_text=generated_text,
                input_length=metadata['input_length'],
                save_path=output_path / "attention_demo.png"
            )
            
            # Save text output
            with open(output_path / "generated_report.txt", "w") as f:
                f.write(f"Generated Report:\n{generated_text}\n")
                f.write(f"\nInput length: {metadata['input_length']} tokens\n")
                f.write(f"Has lateral: {metadata['has_lateral']}\n")
            
            logger.info(f"Demo completed! Check {output_path}/")
            
            if fig:
                plt.close(fig)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise


def create_test_image() -> Image.Image:
    """Create a simple test chest X-ray-like image"""
    image = np.ones((512, 512), dtype=np.uint8) * 100  # Gray background
    
    # Add some structure
    image[100:400, 150:200] = 50   # Left lung area (darker)
    image[100:400, 300:350] = 50   # Right lung area (darker)
    image[250:270, :] = 120        # Ribs
    image[200:220, :] = 120
    image[300:320, :] = 120
    
    return Image.fromarray(image, mode='L')


def main():
    """Demo the lightweight visualizer"""
    
    try:
        # Check memory first
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        logger.info(f"Available RAM: {available_gb:.1f} GB")
        
        if available_gb < 0.5:
            logger.error("Insufficient memory to run MAIRA-2")
            logger.info("Try closing other applications first")
            return False
        
        # Create test image
        logger.info("Creating test image...")
        test_image = create_test_image()
        
        # Initialize visualizer
        logger.info("Initializing lightweight visualizer...")
        visualizer = LightweightMAIRA2Visualizer()
        
        # Run demo
        generated_text = visualizer.quick_demo(
            frontal_image=test_image,
            output_dir="lightweight_output"
        )
        
        logger.info(f"🎉 Success! Generated: {generated_text}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)