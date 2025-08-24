#!/usr/bin/env python3
"""
Test runner script that mimics what VS Code pytest plugin should do

This script verifies that tests can be discovered and run properly
in the uv environment, which is what VS Code will use.
"""

import subprocess
import sys
from pathlib import Path

def run_pytest_commands():
    """Run various pytest commands to verify functionality"""
    
    project_root = Path(__file__).parent
    
    print("Testing VS Code pytest integration with uv environment")
    print("=" * 60)
    
    commands = [
        # Basic test discovery
        {
            "name": "Test Discovery",
            "cmd": ["uv", "run", "pytest", "--collect-only", "-q"],
            "description": "Check that tests can be discovered"
        },
        
        # Run environment tests (fast)
        {
            "name": "Environment Tests", 
            "cmd": ["uv", "run", "pytest", "tests/test_environment.py", "-v"],
            "description": "Run quick environment validation tests"
        },
        
        # Run unit tests only (skip slow tests)
        {
            "name": "Unit Tests Only",
            "cmd": ["uv", "run", "pytest", "-m", "unit", "-v"],
            "description": "Run only unit tests (fast)"
        },
        
        # Test markers work
        {
            "name": "Test Markers",
            "cmd": ["uv", "run", "pytest", "--markers"],
            "description": "Show available test markers"
        },
        
        # Simulate VS Code pytest run
        {
            "name": "VS Code Style Run",
            "cmd": ["uv", "run", "pytest", "tests", "-v", "--tb=short"],
            "description": "Simulate VS Code pytest plugin execution"
        }
    ]
    
    results = []
    
    for test in commands:
        print(f"\n{test['name']}")
        print("-" * 40)
        print(f"Description: {test['description']}")
        print(f"Command: {' '.join(test['cmd'])}")
        print()
        
        try:
            result = subprocess.run(
                test['cmd'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ SUCCESS")
                results.append(("PASS", test['name']))
            else:
                print(f"❌ FAILED (exit code: {result.returncode})")
                results.append(("FAIL", test['name']))
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
            
            # Show some output for context
            if result.stdout:
                lines = result.stdout.split('\n')
                # Show first few and last few lines
                if len(lines) > 10:
                    print('\n'.join(lines[:5]))
                    print("...")
                    print('\n'.join(lines[-3:]))
                else:
                    print(result.stdout)
                    
        except subprocess.TimeoutExpired:
            print("⏱️  TIMEOUT")
            results.append(("TIMEOUT", test['name']))
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append(("ERROR", test['name']))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for status, name in results:
        status_symbol = {
            "PASS": "✅",
            "FAIL": "❌", 
            "TIMEOUT": "⏱️",
            "ERROR": "💥"
        }.get(status, "❓")
        
        print(f"{status_symbol} {name}: {status}")
    
    # VS Code integration notes
    print("\n" + "=" * 60)
    print("VS CODE INTEGRATION NOTES")
    print("=" * 60)
    print("If all tests above passed, VS Code pytest plugin should work with:")
    print("1. Python interpreter: ./.venv/bin/python")
    print("2. Pytest path: ./.venv/bin/pytest")
    print("3. Working directory: project root")
    print("4. Test directory: ./tests")
    print()
    print("In VS Code:")
    print("- Open Command Palette (Ctrl+Shift+P)")
    print("- Type 'Python: Configure Tests'")
    print("- Select 'pytest'")
    print("- Select './tests' as test directory")
    print("- Tests should appear in Test Explorer")

def main():
    """Main function"""
    try:
        run_pytest_commands()
        return True
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)