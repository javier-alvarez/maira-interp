#!/usr/bin/env python3
"""
MAIRA-2 Attention Visualization Tool

This tool visualizes attention maps for every output token in MAIRA-2, 
showing how the model attends to different parts of the input (frontal CXR, 
lateral view, prior image, or text) during generation.

MAIRA-2 Architecture:
- Vision Encoder: RAD-DINO-MAIRA-2 (DINOv2-based) - frozen, 12 layers, 12 heads each
- Multimodal Projector: 4-layer MLP with GELU activation 
- Language Model: Vicuna-7B-v1.5 (32 layers, 32 heads each)
- Image sequence length: 576 tokens per image
- Supports multiple inputs: frontal CXR, lateral view, prior frontal, text sections
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import json
import os

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAIRA2AttentionVisualizer:
    """Visualizes attention patterns in MAIRA-2 for every output token"""
    
    def __init__(self, model_path: str = "microsoft/maira-2", token: Optional[str] = None):
        """Initialize the visualizer with MAIRA-2 model"""
        logger.info(f"Loading MAIRA-2 model from {model_path}")
        
        # Get token from parameter, environment variable, or None
        auth_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
        
        if auth_token:
            logger.info("Using provided Hugging Face token for authentication")
        else:
            logger.warning("No Hugging Face token found. You may need to authenticate for gated models like MAIRA-2.")
            logger.info("Get a token from: https://huggingface.co/settings/tokens")
            logger.info("Then set it as: export HF_TOKEN=your_token_here")
        
        # Load model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=auth_token
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, token=auth_token)
        
        # Set to eval mode
        self.model.eval()
        
        # Model architecture info
        self.vision_layers = 12  # DINOv2 has 12 layers
        self.vision_heads = 12   # 12 attention heads per layer
        self.text_layers = 32    # Vicuna-7B has 32 layers  
        self.text_heads = 32     # 32 attention heads per layer
        self.image_seq_length = 576  # Image tokens per image
        
        logger.info("MAIRA-2 model loaded successfully")
        
    def hook_attention_weights(self) -> Dict[str, List]:
        """Add forward hooks to capture attention weights from all layers"""
        attention_weights = {
            'vision_attention': [],
            'text_attention': []
        }
        
        def vision_attention_hook(module, input, output):
            # DINOv2 attention output: (batch_size, seq_len, hidden_size)
            # We need to capture the attention weights from the attention mechanism
            if hasattr(module, 'attention') and hasattr(module.attention, 'dropout'):
                # Get attention weights if available
                if hasattr(module.attention, 'attention_probs'):
                    attention_weights['vision_attention'].append(
                        module.attention.attention_probs.detach().cpu()
                    )
        
        def text_attention_hook(module, input, output):
            # LLaMA attention weights: (batch_size, num_heads, seq_len, seq_len)
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights['text_attention'].append(
                    output.attentions.detach().cpu()
                )
        
        # Register hooks for vision tower (DINOv2)
        vision_hooks = []
        if hasattr(self.model, 'vision_tower'):
            for layer in self.model.vision_tower.encoder.layer:
                if hasattr(layer, 'attention'):
                    hook = layer.attention.register_forward_hook(vision_attention_hook)
                    vision_hooks.append(hook)
        
        # Register hooks for language model (Vicuna/LLaMA)
        text_hooks = []
        if hasattr(self.model, 'language_model'):
            for layer in self.model.language_model.model.layers:
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(text_attention_hook)
                    text_hooks.append(hook)
        
        return attention_weights, vision_hooks + text_hooks
    
    def process_inputs(
        self, 
        frontal_image: Image.Image,
        lateral_image: Optional[Image.Image] = None,
        prior_frontal: Optional[Image.Image] = None,
        indication: Optional[str] = None,
        technique: Optional[str] = None,
        comparison: Optional[str] = None,
        prior_report: Optional[str] = None,
        get_grounding: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """Process inputs for MAIRA-2 and return input tensors with metadata"""
        
        # Format inputs using the processor
        processed = self.processor.format_and_preprocess_reporting_input(
            current_frontal=frontal_image,
            current_lateral=lateral_image,
            prior_frontal=prior_frontal,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=prior_report,
            get_grounding=get_grounding,
            return_tensors="pt"
        )
        
        # Move to model device
        processed = processed.to(self.model.device)
        
        # Create metadata about input structure
        input_metadata = {
            'has_frontal': True,
            'has_lateral': lateral_image is not None,
            'has_prior': prior_frontal is not None,
            'input_ids': processed['input_ids'],
            'pixel_values': processed.get('pixel_values'),
            'num_images': 1 + (1 if lateral_image else 0) + (1 if prior_frontal else 0),
            'image_positions': self._get_image_token_positions(processed['input_ids'])
        }
        
        return processed, input_metadata
    
    def _get_image_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        """Find positions of image tokens in the input sequence"""
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        positions = []
        
        for i, token_id in enumerate(input_ids[0]):
            if token_id == image_token_id:
                positions.append(i)
                
        return positions
    
    def generate_with_attention(
        self,
        processed_inputs: torch.Tensor,
        max_new_tokens: int = 300,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate tokens while capturing attention weights"""
        
        # Hook attention weights
        attention_weights, hooks = self.hook_attention_weights()
        
        try:
            # Generate with attention output
            with torch.no_grad():
                outputs = self.model.generate(
                    **processed_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    output_attentions=return_attention,
                    return_dict_in_generate=True,
                    do_sample=False  # Use greedy decoding for consistent attention
                )
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
                
            return outputs, attention_weights
            
        except Exception as e:
            # Clean up hooks in case of error
            for hook in hooks:
                hook.remove()
            raise e
    
    def visualize_token_attention(
        self,
        attention_weights: Dict,
        input_metadata: Dict,
        output_tokens: torch.Tensor,
        token_idx: int,
        layer_idx: int = -1,  # -1 for last layer
        head_idx: Optional[int] = None,  # None for average across heads
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention for a specific output token"""
        
        if not attention_weights['text_attention']:
            logger.warning("No text attention weights captured")
            return None
            
        # Get attention weights for the specified layer
        if layer_idx == -1:
            layer_idx = len(attention_weights['text_attention']) - 1
            
        attn = attention_weights['text_attention'][layer_idx]  # (batch, heads, seq_len, seq_len)
        
        if attn is None or attn.numel() == 0:
            logger.warning(f"No attention weights for layer {layer_idx}")
            return None
            
        # Focus on the specific output token
        token_attention = attn[0, :, token_idx, :]  # (heads, input_seq_len)
        
        # Average across heads if not specified
        if head_idx is None:
            token_attention = token_attention.mean(dim=0)  # (input_seq_len,)
        else:
            token_attention = token_attention[head_idx]  # (input_seq_len,)
            
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attention for Token {token_idx} (Layer {layer_idx})', fontsize=16)
        
        # 1. Full attention heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(token_attention.unsqueeze(0).numpy(), cmap='Blues', aspect='auto')
        ax1.set_title('Full Attention Pattern')
        ax1.set_xlabel('Input Token Position')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Image attention (if images present)
        if input_metadata.get('pixel_values') is not None:
            ax2 = axes[0, 1]
            image_positions = input_metadata['image_positions']
            
            if image_positions:
                # Extract attention to image tokens
                image_attention = []
                for pos in image_positions:
                    # Each image has 576 tokens
                    start_pos = pos
                    end_pos = min(pos + self.image_seq_length, len(token_attention))
                    img_attn = token_attention[start_pos:end_pos]
                    image_attention.append(img_attn.numpy())
                
                # Visualize first image attention as 24x24 grid (576 = 24*24)
                if len(image_attention) > 0:
                    img_attn_2d = image_attention[0].reshape(24, 24)
                    im2 = ax2.imshow(img_attn_2d, cmap='Reds')
                    ax2.set_title('Attention to Frontal CXR')
                    plt.colorbar(im2, ax=ax2)
                else:
                    ax2.text(0.5, 0.5, 'No Image Tokens', ha='center', va='center')
                    ax2.set_title('Image Attention')
            else:
                ax2.text(0.5, 0.5, 'No Image Tokens Found', ha='center', va='center')
                ax2.set_title('Image Attention')
        
        # 3. Text attention distribution
        ax3 = axes[1, 0]
        text_positions = [i for i in range(len(token_attention)) 
                         if i not in input_metadata.get('image_positions', [])]
        if text_positions:
            text_attention = token_attention[text_positions].numpy()
            ax3.bar(range(len(text_attention)), text_attention)
            ax3.set_title('Attention to Text Tokens')
            ax3.set_xlabel('Text Token Position')
            ax3.set_ylabel('Attention Weight')
        else:
            ax3.text(0.5, 0.5, 'No Text Tokens', ha='center', va='center')
            ax3.set_title('Text Attention')
        
        # 4. Attention statistics
        ax4 = axes[1, 1]
        stats_text = f"""
        Token Index: {token_idx}
        Layer: {layer_idx}
        Head: {'Average' if head_idx is None else head_idx}
        
        Attention Stats:
        Max: {token_attention.max():.4f}
        Min: {token_attention.min():.4f}
        Mean: {token_attention.mean():.4f}
        Std: {token_attention.std():.4f}
        
        Input Structure:
        Images: {input_metadata['num_images']}
        Image Positions: {input_metadata['image_positions']}
        Sequence Length: {len(token_attention)}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        ax4.set_title('Attention Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
            
        return fig
    
    def generate_attention_pngs(
        self,
        frontal_image: Image.Image,
        lateral_image: Optional[Image.Image] = None,
        prior_frontal: Optional[Image.Image] = None,
        indication: Optional[str] = None,
        technique: Optional[str] = None,
        comparison: Optional[str] = None,
        prior_report: Optional[str] = None,
        get_grounding: bool = False,
        max_new_tokens: int = 100,
        output_dir: str = "attention_outputs",
        visualize_every_n: int = 5  # Visualize every 5th token to avoid too many images
    ):
        """Generate PNG visualizations for attention maps of output tokens"""
        
        logger.info("Processing inputs...")
        processed_inputs, input_metadata = self.process_inputs(
            frontal_image=frontal_image,
            lateral_image=lateral_image,
            prior_frontal=prior_frontal,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=prior_report,
            get_grounding=get_grounding
        )
        
        logger.info("Generating tokens with attention tracking...")
        outputs, attention_weights = self.generate_with_attention(
            processed_inputs, max_new_tokens=max_new_tokens
        )
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get generated tokens
        prompt_length = processed_inputs['input_ids'].shape[-1]
        generated_tokens = outputs.sequences[0][prompt_length:]
        
        # Decode tokens for reference
        decoded_tokens = []
        for i, token_id in enumerate(generated_tokens):
            token_text = self.processor.tokenizer.decode([token_id], skip_special_tokens=False)
            decoded_tokens.append(f"{i}: '{token_text}' (ID: {token_id})")
        
        # Save token information
        with open(output_path / "generated_tokens.txt", "w") as f:
            f.write("Generated Tokens:\n")
            f.write("\n".join(decoded_tokens))
            
        # Save input metadata
        metadata_to_save = {
            "num_images": input_metadata["num_images"],
            "has_frontal": input_metadata["has_frontal"],
            "has_lateral": input_metadata["has_lateral"], 
            "has_prior": input_metadata["has_prior"],
            "image_positions": input_metadata["image_positions"],
            "sequence_length": processed_inputs['input_ids'].shape[-1],
            "generated_tokens_count": len(generated_tokens)
        }
        
        with open(output_path / "input_metadata.json", "w") as f:
            json.dump(metadata_to_save, f, indent=2)
        
        logger.info(f"Generating attention visualizations for {len(generated_tokens)} tokens...")
        
        # Generate visualizations for selected tokens
        tokens_to_visualize = range(0, len(generated_tokens), visualize_every_n)
        
        for token_idx in tqdm(tokens_to_visualize, desc="Creating attention maps"):
            # Adjust token index to account for full sequence (prompt + generated)
            full_token_idx = prompt_length + token_idx
            
            try:
                fig = self.visualize_token_attention(
                    attention_weights=attention_weights,
                    input_metadata=input_metadata,
                    output_tokens=outputs.sequences[0],
                    token_idx=full_token_idx,
                    save_path=output_path / f"attention_token_{token_idx:03d}.png"
                )
                
                if fig:
                    plt.close(fig)  # Close to free memory
                    
            except Exception as e:
                logger.warning(f"Failed to create visualization for token {token_idx}: {e}")
                
        logger.info(f"Attention visualizations saved to {output_path}")
        
        # Generate final report text
        final_text = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        
        with open(output_path / "generated_report.txt", "w") as f:
            f.write("Generated Report:\n")
            f.write(final_text)
            
        logger.info(f"Generated report: {final_text}")
        
        return output_path, final_text


def main():
    """Example usage of the MAIRA-2 attention visualizer"""
    
    # Initialize visualizer
    visualizer = MAIRA2AttentionVisualizer()
    
    # Example: Load a sample chest X-ray
    # You can replace this with your own image
    try:
        import requests
        
        # Download sample image from IU-Xray dataset (CC license)
        frontal_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
        lateral_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-2001.png"
        
        logger.info("Downloading sample images...")
        frontal_response = requests.get(frontal_url, headers={"User-Agent": "MAIRA-2-Attention-Visualizer"})
        lateral_response = requests.get(lateral_url, headers={"User-Agent": "MAIRA-2-Attention-Visualizer"})
        
        frontal_image = Image.open(frontal_response.raw)
        lateral_image = Image.open(lateral_response.raw)
        
        # Generate attention visualizations
        output_dir, generated_report = visualizer.generate_attention_pngs(
            frontal_image=frontal_image,
            lateral_image=lateral_image,
            indication="Dyspnea.",
            technique="PA and lateral views of the chest.",
            comparison="None.",
            max_new_tokens=50,  # Generate fewer tokens for demo
            visualize_every_n=3  # Visualize every 3rd token
        )
        
        print(f"\nGenerated report: {generated_report}")
        print(f"Attention visualizations saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        
        # Fallback: create example with placeholder
        print("Could not download sample images. Please provide your own images.")
        print("Usage example:")
        print("""
        from PIL import Image
        
        # Load your images
        frontal = Image.open("path/to/frontal.png")
        lateral = Image.open("path/to/lateral.png")  # optional
        
        # Create visualizer
        viz = MAIRA2AttentionVisualizer()
        
        # Generate attention maps
        output_dir, report = viz.generate_attention_pngs(
            frontal_image=frontal,
            lateral_image=lateral,
            indication="Your indication text",
            technique="PA and lateral views",
            comparison="Comparison text"
        )
        """)


if __name__ == "__main__":
    main()