#!/usr/bin/env python3
"""
Setup script for Hugging Face authentication for MAIRA-2
"""

import os
import sys
from pathlib import Path

def setup_hf_authentication():
    """Interactive setup for Hugging Face authentication"""
    
    print("MAIRA-2 Authentication Setup")
    print("=" * 40)
    print()
    
    # Check if already authenticated
    existing_token = (os.getenv("HF_TOKEN") or 
                     os.getenv("HUGGING_FACE_HUB_TOKEN") or 
                     os.getenv("HF_HUB_TOKEN"))
    
    if existing_token:
        print("✅ Hugging Face token already found in environment!")
        print("You should be able to access MAIRA-2.")
        return
    
    print("🔐 MAIRA-2 is a gated model that requires authentication.")
    print()
    print("Steps to get access:")
    print("1. Go to: https://huggingface.co/microsoft/maira-2")
    print("2. Click 'Request access' and agree to the terms")
    print("3. Wait for approval (usually automatic)")
    print("4. Go to: https://huggingface.co/settings/tokens")
    print("5. Create a new token with 'Read' permissions")
    print()
    
    choice = input("Do you want to set up authentication now? (y/n): ").lower().strip()
    
    if choice not in ['y', 'yes']:
        print("Setup cancelled. You can run this script again later.")
        return
    
    print()
    print("Choose an authentication method:")
    print("1. Set environment variable (recommended)")
    print("2. Create .env file (for this project only)")
    print("3. Use huggingface-cli login")
    
    method = input("Enter choice (1-3): ").strip()
    
    if method == "1":
        setup_env_variable()
    elif method == "2":
        setup_env_file()
    elif method == "3":
        setup_cli_login()
    else:
        print("Invalid choice. Please run the script again.")

def setup_env_variable():
    """Setup environment variable method"""
    print()
    print("Setting up environment variable...")
    print()
    
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided. Setup cancelled.")
        return
    
    # Detect shell
    shell = os.getenv("SHELL", "/bin/bash")
    
    if "zsh" in shell:
        rc_file = Path.home() / ".zshrc"
    elif "fish" in shell:
        rc_file = Path.home() / ".config/fish/config.fish"
        export_cmd = f"set -gx HF_TOKEN {token}"
    else:  # bash or others
        rc_file = Path.home() / ".bashrc"
    
    if "fish" not in shell:
        export_cmd = f"export HF_TOKEN={token}"
    
    print(f"Adding to {rc_file}:")
    print(f"  {export_cmd}")
    
    confirm = input("Proceed? (y/n): ").lower().strip()
    
    if confirm in ['y', 'yes']:
        try:
            with open(rc_file, "a") as f:
                f.write(f"\n# Hugging Face token for MAIRA-2\n")
                f.write(f"{export_cmd}\n")
            
            print(f"✅ Token added to {rc_file}")
            print("🔄 Please run: source ~/.bashrc (or restart your terminal)")
            print("   Then try running the visualizer again.")
            
        except Exception as e:
            print(f"❌ Error writing to {rc_file}: {e}")
    else:
        print("Setup cancelled.")

def setup_env_file():
    """Setup .env file method"""
    print()
    print("Setting up .env file...")
    print()
    
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided. Setup cancelled.")
        return
    
    env_file = Path(".env")
    
    try:
        with open(env_file, "w") as f:
            f.write(f"# Hugging Face Authentication\n")
            f.write(f"HF_TOKEN={token}\n")
        
        print(f"✅ Token saved to {env_file}")
        print("🔄 The visualizer will automatically load this token.")
        
        # Add .env to .gitignore if it exists
        gitignore = Path(".gitignore")
        if gitignore.exists():
            with open(gitignore, "r") as f:
                content = f.read()
            
            if ".env" not in content:
                with open(gitignore, "a") as f:
                    f.write("\n# Environment variables\n.env\n")
                print("✅ Added .env to .gitignore")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def setup_cli_login():
    """Setup using huggingface-cli"""
    print()
    print("Setting up using huggingface-cli...")
    print()
    
    try:
        import subprocess
        
        # Check if huggingface-hub is installed
        result = subprocess.run(["huggingface-cli", "--version"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Installing huggingface-hub...")
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub[cli]"], 
                         check=True)
        
        print("Running: huggingface-cli login")
        subprocess.run(["huggingface-cli", "login"])
        
        print("✅ Authentication complete!")
        print("🔄 Try running the visualizer again.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running huggingface-cli: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_authentication():
    """Test if authentication works"""
    print()
    print("Testing authentication...")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        # Try to get info about MAIRA-2
        info = api.model_info("microsoft/maira-2")
        
        print("✅ Authentication successful!")
        print(f"✅ Access to MAIRA-2 confirmed")
        print(f"   Model ID: {info.modelId}")
        print(f"   Last modified: {info.lastModified}")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your token and permissions.")
        return False

if __name__ == "__main__":
    setup_hf_authentication()
    
    # Test authentication if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_authentication()