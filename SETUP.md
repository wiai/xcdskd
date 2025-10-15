# Setup Instructions

## Quick Start

1. Run the WinPython installer:
   ```cmd
   python\install_WinPython.bat
   ```

2. This will:
   - Download WinPython 3.13.7.0slim
   - Extract to `python/WPy64-31370/`
   - Install uv package manager
   - Install aloe package with dependencies

3. Activate the environment:
   ```
   python\run.cmd
   ```

## Manual Installation

If you already have WinPython installed:

```cmd
cd python
call WPy64-31370\scripts\env.bat
cd ..
pip install uv
uv pip install -e .
```

## Optional Dependencies

Install scientific packages:
```
uv pip install -e .[scientific]
```

Install development tools:
```
uv pip install -e .[dev]
```

## Managing WinPython Versions

The project uses a configuration system to manage WinPython versions. This makes it easy to switch between different Python versions without editing multiple script files.

### Version Configuration

**Main Configuration File:** `python/version_config.txt`

To switch WinPython versions:

1. **Edit the configuration:**
   ```txt
   # In python/version_config.txt, change this line:
   WINPYTHON_VERSION=WPy64-31370
   ```

2. **Check available versions:**
   ```cmd
   dir python\WPy* /b
   ```

3. **Verify the configuration:**
   ```cmd
   python\get_python_version.bat
   ```

### Version Management Scripts

- **`python/get_python_version.bat`** - Reads configuration and validates version exists
- **`python/setup_python.bat`** - Downloads and installs the configured WinPython version
- **All other scripts** - Automatically use the configured version

### Example Version Switches

```txt
# For Python 3.13.7.0 (default)
WINPYTHON_VERSION=WPy64-31370

# For Python 3.12.1.0
WINPYTHON_VERSION=WPy64-312101

# For Python 3.11.9.0
WINPYTHON_VERSION=WPy64-31190
```

### Troubleshooting Version Issues

**If you see warnings about missing versions:**
1. Run `python\install_WinPython.bat` to install the configured version
2. Or edit `python/version_config.txt` to use an existing version
3. Check available versions with `dir python\WPy* /b`

**Version not detected?**
- Ensure the version follows the pattern `WPy64-XXXXX`
- Check that the WinPython directory exists in `python/`
- Verify the configuration file format

## Old Setup (Deprecated)

The conda environment setup is deprecated. Use the WinPython setup above.