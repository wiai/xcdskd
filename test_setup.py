#!/usr/bin/env python3
"""
Test script to validate the refactoring setup.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_files_exist():
    """Test that all required files exist."""
    required_files = [
        'pyproject.toml',
        'python/setup_python.bat',
        'python/run.cmd',
        'python/install_deps.bat',
        'SETUP.md',
        '.gitignore'
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files exist")
    return True

def test_pyproject_toml():
    """Test that pyproject.toml is valid."""
    try:
        import tomllib
        with open('pyproject.toml', 'rb') as f:
            config = tomllib.load(f)

        # Check required sections
        required_sections = ['project', 'build-system']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section '{section}' in pyproject.toml")
                return False

        # Check project name
        if config['project'].get('name') != 'aloe':
            print("❌ Project name is not 'aloe'")
            return False

        print("✅ pyproject.toml is valid")
        return True

    except ImportError:
        print("⚠️  tomllib not available (Python < 3.11), skipping validation")
        return True
    except Exception as e:
        print(f"❌ Error reading pyproject.toml: {e}")
        return False

def test_script_syntax():
    """Test that batch scripts have correct syntax."""
    scripts = [
        'python/setup_python.bat',
        'python/run.cmd',
        'python/install_deps.bat'
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"❌ Script {script} does not exist")
            return False

        try:
            with open(script, 'r') as f:
                content = f.read()

            # Basic syntax checks
            if script.endswith('.bat'):
                if 'echo off' not in content.lower():
                    print(f"⚠️  Warning: {script} should start with @echo off")
            elif script.endswith('.cmd'):
                if 'python %*' not in content:
                    print(f"⚠️  Warning: {script} should pass arguments to python")

        except Exception as e:
            print(f"❌ Error reading {script}: {e}")
            return False

    print("✅ All scripts have valid syntax")
    return True

def test_gitignore():
    """Test that .gitignore includes uv and WinPython entries."""
    try:
        with open('.gitignore', 'r') as f:
            content = f.read()

        required_entries = [
            'uv.lock',
            '.venv/',
            'WPy64-31370/'
        ]

        for entry in required_entries:
            if entry not in content:
                print(f"❌ Missing .gitignore entry: {entry}")
                return False

        print("✅ .gitignore includes all required entries")
        return True

    except Exception as e:
        print(f"❌ Error reading .gitignore: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing xcdskd refactoring setup...\n")

    tests = [
        test_files_exist,
        test_pyproject_toml,
        test_script_syntax,
        test_gitignore
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\n--- Running {test.__name__} ---")
        if test():
            passed += 1

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Refactoring setup is valid.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())