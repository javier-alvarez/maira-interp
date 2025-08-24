#!/usr/bin/env python3
"""
Simple test for Hugging Face authentication
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

def test_token():
    """Test if we have a valid token"""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
    
    if not token:
        print("❌ No Hugging Face token found in environment variables")
        print("Check your .env file or set HF_TOKEN environment variable")
        return False
    
    print(f"✅ Found Hugging Face token: {token[:10]}...")
    
    # Test access to MAIRA-2
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        info = api.model_info("microsoft/maira-2")
        
        print("✅ Successfully authenticated with Hugging Face!")
        print(f"✅ Access to MAIRA-2 confirmed")
        print(f"   Model ID: {info.modelId}")
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check:")
        print("1. Your token is valid")
        print("2. You have been granted access to MAIRA-2")
        print("3. Go to: https://huggingface.co/microsoft/maira-2")
        return False

if __name__ == "__main__":
    test_token()