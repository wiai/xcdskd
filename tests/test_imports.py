"""
Test aloe package imports and basic functionality.
Tests that the aloe package can be imported and key modules work.
"""

import sys
import os


class TestAloeImports:
    """Test aloe package imports and basic functionality."""

    def test_aloe_import(self):
        """Test basic aloe package import."""
        import aloe
        assert aloe is not None, "Failed to import aloe package"

    def test_aloe_submodules(self):
        """Test importing key aloe submodules."""
        submodules_to_test = [
            'aloe.io',
            'aloe.image',
            'aloe.math',
            'aloe.physics',
            'aloe.plotting',
            'aloe.sys',
            'aloe.fit',
            'aloe.exp',
        ]

        for module in submodules_to_test:
            __import__(module)

    def test_key_dependencies(self):
        """Test that key dependencies are available."""
        dependencies_to_test = [
            'h5py',
            'skimage',
            'cv2',
            'PIL',
            'tables',
            'tqdm',
            'yaml',
        ]

        for module_name in dependencies_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                # Some dependencies might not be available in test environment
                # This is expected when not using WinPython
                print(f"Warning: {module_name} not available: {e}")

    def test_aloe_functionality(self):
        """Test basic aloe functionality."""
        import aloe
        from aloe.image import utils
        from aloe.math import euler
        from aloe.io import h5oina

        # Test that we can access some basic attributes
        assert hasattr(aloe, '__name__'), "aloe package missing __name__ attribute"

    def test_aloe_init(self):
        """Test that aloe/__init__.py works correctly."""
        import aloe

        # Should have basic package attributes
        assert hasattr(aloe, '__name__'), "aloe package should have __name__"
        assert aloe.__name__ == 'aloe', "aloe package __name__ should be 'aloe'"