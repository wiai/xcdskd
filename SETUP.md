# Setup Instructions

## Quick Start

1. Run the WinPython installer:
   ```
   python\setup_python.bat
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

```batch
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

## Old Setup (Deprecated)

The conda environment setup is deprecated. Use the WinPython setup above.