"""
Test setup validation for xcdskd refactoring.
Tests that all required files exist and have correct content.
"""

import os
import sys
from pathlib import Path


class TestSetupValidation:
    """Test that all required files exist and are properly configured."""

    def test_required_files_exist(self):
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

        assert len(missing_files) == 0, f"Missing files: {missing_files}"

    def test_pyproject_toml_exists_and_valid(self):
        """Test that pyproject.toml exists and has required sections."""
        assert os.path.exists('pyproject.toml'), "pyproject.toml does not exist"

        # Basic validation that file is not empty and has content
        with open('pyproject.toml', 'r') as f:
            content = f.read()

        assert len(content) > 0, "pyproject.toml is empty"
        assert 'name = "aloe"' in content, "Project name not found in pyproject.toml"
        assert '[project]' in content, "Missing [project] section"
        assert '[build-system]' in content, "Missing [build-system] section"

    def test_batch_scripts_exist(self):
        """Test that batch scripts exist and have basic content."""
        scripts = [
            'python/setup_python.bat',
            'python/run.cmd',
            'python/install_deps.bat'
        ]

        for script in scripts:
            assert os.path.exists(script), f"Script {script} does not exist"

            with open(script, 'r') as f:
                content = f.read()

            assert len(content) > 0, f"Script {script} is empty"
            # Scripts should either start with @echo off or :: (comment for .cmd files)
            assert ('@echo off' in content.lower() or
                    content.strip().startswith('::')), f"Script {script} should start with @echo off or ::"

    def test_setup_md_exists(self):
        """Test that SETUP.md documentation exists."""
        assert os.path.exists('SETUP.md'), "SETUP.md does not exist"

        with open('SETUP.md', 'r') as f:
            content = f.read()

        assert len(content) > 0, "SETUP.md is empty"
        assert 'python\\setup_python.bat' in content, "SETUP.md should mention setup script"

    def test_gitignore_updated(self):
        """Test that .gitignore includes uv and WinPython entries."""
        assert os.path.exists('.gitignore'), ".gitignore does not exist"

        with open('.gitignore', 'r') as f:
            content = f.read()

        required_entries = [
            'uv.lock',
            '.venv/',
            'WPy64-31370/'
        ]

        for entry in required_entries:
            assert entry in content, f"Missing .gitignore entry: {entry}"