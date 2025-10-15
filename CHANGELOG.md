# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.0] - 2025-10-15

### 🚀 Major Changes

#### Package Management & Build System Modernization
- **Removed legacy setup.py** - Fully migrated to pyproject.toml (PEP 621 standard)
- **Replaced conda with uv package manager** - Faster, more reliable dependency resolution
- **Updated to WinPython 3.13.7.0slim** - Portable Python environment (upgraded from 3.12.1)
- **Automated setup scripts** - One-click installation with `python/setup_python.bat`

#### Testing Infrastructure
- **Added pytest framework** - Proper test discovery and execution
- **Organized tests in tests/ directory** - Standard Python project structure
- **Added pytest configuration** - Configured in pyproject.toml with proper discovery patterns
- **Removed GUI dependencies from tests** - Cleaner test environment

#### Documentation Updates
- **Created comprehensive SETUP.md** - New installation and setup guide
- **Updated README.rst** - Modern WinPython-based installation instructions
- **Enhanced .gitignore** - Added uv, pytest, and WinPython-specific exclusions

#### Script Updates
- **Updated doc/run_notebook.cmd** - Now uses WPy64-31370 and uv package manager
- **Updated apps/tsl2h5oina/run.cmd** - Updated to use new WinPython version

#### Version Management System
- **Created python/version_config.txt** - Centralized WinPython version configuration
- **Created python/get_python_version.bat** - Helper script for version management
- **Updated all scripts** - Now use configuration system instead of hardcoded versions
- **Easy version switching** - Change version in one file, affects all scripts

#### Download Method Improvements
- **Fixed install_WinPython.bat** - Replaced PowerShell with curl for downloading
- **Added fallback method** - Uses bitsadmin if curl fails
- **No Python dependency** - Download method doesn't require Python

### 🔧 Technical Improvements

#### Dependencies Management
- **Minimal core dependencies** - Let uv handle transitive dependencies for better compatibility
- **Optional dependency groups** - Scientific and development packages separated
- **Fixed package names** - Corrected "pytables" to "tables" in dependencies

#### Development Workflow
- **Faster package installation** - uv provides significant speed improvements
- **Isolated Python environment** - No system Python conflicts
- **Automated testing** - Run tests with `python -m pytest tests/`

### 📋 Migration Summary

| Component | Before | After |
|-----------|--------|-------|
| **Package Config** | setup.py | pyproject.toml (only) |
| **Package Manager** | pip (manual) | uv (automated) |
| **Python Version** | 3.12.1 | 3.13.7.0slim |
| **Environment** | Conda | WinPython (portable) |
| **Testing** | Manual | pytest framework |
| **Setup Process** | Multi-step manual | One-click automated |

### ✅ Verification

- **All tests passing** (10/10 tests in pytest)
- **Package installs correctly** with uv
- **All imports functional** - aloe package and dependencies
- **Documentation updated** - Clear setup instructions
- **Backward compatibility** - Existing code continues to work

## [0.1.0] - 2024-??-??

### Added
- Initial release of xcdskd package
- Basic aloe package functionality
- Conda environment setup
- Initial documentation and examples