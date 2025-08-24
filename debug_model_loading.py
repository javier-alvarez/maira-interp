#!/usr/bin/env python3
"""
Debug script for MAIRA-2 model loading issues
"""

import os
import sys
import logging
import traceback
from pathlib import Path
import torch
import psutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check available system resources"""
    logger.info("=== SYSTEM RESOURCE CHECK ===")
    
    # Memory
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    logger.info(f"Memory usage: {memory.percent}%")
    
    # Disk space
    disk = psutil.disk_usage('/')
    logger.info(f"Total disk: {disk.total / (1024**3):.1f} GB")
    logger.info(f"Free disk: {disk.free / (1024**3):.1f} GB")
    
    # CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"  Device {i}: {props.name}, {memory_gb:.1f} GB VRAM")
            
            # Check current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"    Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        logger.info("CUDA available: No")

def check_authentication():
    """Check Hugging Face authentication"""
    logger.info("=== AUTHENTICATION CHECK ===")
    
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
    
    if not token:
        logger.error("No Hugging Face token found!")
        return False
    
    logger.info(f"Found token: {token[:10]}...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.model_info("microsoft/maira-2")
        logger.info("✅ Authentication successful")
        try:
            total_size = sum(f.size for f in info.siblings if f.size is not None)
            logger.info(f"Model size: ~{total_size / (1024**3):.1f} GB")
        except (AttributeError, TypeError):
            logger.info("Model size: Unable to calculate")
        return True
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False

def debug_model_loading():
    """Debug model loading with detailed progress"""
    logger.info("=== MODEL LOADING DEBUG ===")
    
    # Get authentication token
    auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
    
    try:
        # Import transformers with debug logging
        import transformers
        transformers.logging.set_verbosity_debug()
        
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        
        logger.info("Step 1: Loading model configuration...")
        config = AutoConfig.from_pretrained(
            "microsoft/maira-2", 
            trust_remote_code=True,
            token=auth_token
        )
        logger.info(f"✅ Config loaded: {config.model_type}")
        logger.info(f"Text config: {config.text_config.model_type}")
        logger.info(f"Vision config: {config.vision_config.model_type}")
        
        logger.info("Step 2: Loading processor...")
        processor = AutoProcessor.from_pretrained(
            "microsoft/maira-2", 
            trust_remote_code=True, 
            token=auth_token
        )
        logger.info("✅ Processor loaded")
        
        logger.info("Step 3: Checking model download...")
        # First, just try to get model info without loading
        from huggingface_hub import snapshot_download
        
        logger.info("Downloading model files...")
        cache_dir = snapshot_download(
            repo_id="microsoft/maira-2",
            token=auth_token,
            local_files_only=False  # Allow download
        )
        logger.info(f"✅ Model files downloaded to: {cache_dir}")
        
        # Check available memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated(0) / (1024**3)
            logger.info(f"GPU memory before loading: {memory_before:.2f} GB")
        
        logger.info("Step 4: Loading model (this may take a while)...")
        logger.info("Using device_map='auto' for automatic device placement")
        
        # Try loading with different strategies
        device_map_strategies = [
            "auto",  # Automatic placement
            {"": 0} if torch.cuda.is_available() else "cpu",  # Single device
            "cpu"  # CPU fallback
        ]
        
        model = None
        for i, device_map in enumerate(device_map_strategies):
            try:
                logger.info(f"Attempt {i+1}: device_map={device_map}")
                
                model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/maira-2",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
                    device_map=device_map,
                    token=auth_token,
                    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                    offload_folder="./offload" if device_map == "auto" else None
                )
                logger.info(f"✅ Model loaded successfully with device_map={device_map}")
                break
                
            except Exception as e:
                logger.error(f"Failed with device_map={device_map}: {e}")
                if i == len(device_map_strategies) - 1:
                    raise  # Re-raise on last attempt
                continue
        
        if model is None:
            raise RuntimeError("Failed to load model with any device mapping strategy")
        
        # Check memory usage after loading
        if torch.cuda.is_available() and hasattr(model, 'hf_device_map'):
            for device_id in set(model.hf_device_map.values()):
                if isinstance(device_id, int):
                    memory_after = torch.cuda.memory_allocated(device_id) / (1024**3)
                    logger.info(f"GPU {device_id} memory after loading: {memory_after:.2f} GB")
        
        logger.info("✅ Model loading completed successfully!")
        logger.info(f"Model device map: {getattr(model, 'hf_device_map', 'Not available')}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("Model loading interrupted by user (Ctrl+C)")
        return False
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory: {e}")
        logger.error("Try reducing batch size or using CPU")
        return False
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def suggest_solutions():
    """Suggest solutions based on system resources"""
    logger.info("=== SUGGESTIONS ===")
    
    memory = psutil.virtual_memory()
    
    if memory.available < 16 * (1024**3):  # Less than 16GB available
        logger.warning("Low system memory detected")
        logger.info("Solutions:")
        logger.info("1. Close other applications to free RAM")
        logger.info("2. Use model offloading: device_map='auto'")
        logger.info("3. Use CPU-only mode (slower)")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            if memory_gb < 12:
                logger.warning(f"GPU {i} has limited memory: {memory_gb:.1f} GB")
                logger.info("MAIRA-2 requires ~13GB VRAM")
                logger.info("Consider using model sharding or CPU fallback")

def main():
    """Main debug function"""
    logger.info("MAIRA-2 Model Loading Debug")
    logger.info("=" * 50)
    
    # Step 1: Check system resources
    check_system_resources()
    
    # Step 2: Check authentication
    if not check_authentication():
        logger.error("Authentication failed. Cannot proceed.")
        return False
    
    # Step 3: Debug model loading
    success = debug_model_loading()
    
    # Step 4: Suggestions
    suggest_solutions()
    
    if success:
        logger.info("🎉 Model loading debug completed successfully!")
        logger.info("You should now be able to run the attention visualizer")
    else:
        logger.error("❌ Model loading failed. Check the logs above for details.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Debug interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)