import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import re
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image


def organize_grid_files(directory: str) -> Tuple[str, int, int, List[Tuple[int, int, int, int, str]], Optional[Tuple[str, Tuple[int, int]]]]:
    """
    Find and organize files matching pattern: prefix_x<INT>y<INT>.<EXT>
    
    Args:
        directory: Path to directory containing the files
        
    Returns:
        tuple: (extension, nrows, ncols, file_list, image_info)
            - extension: Common file extension
            - nrows: Number of unique rows
            - ncols: Number of unique columns
            - file_list: List of (row_idx, col_idx, row_val, col_val, filename) tuples
                        row_idx and col_idx start from 0
            - image_info: Tuple of (image_type, image_size) where image_size is (height, width)
                         or None if files are not images
    """
    # Pattern to match: anything followed by _x<digits>y<digits>.<extension>
    pattern = re.compile(r'^(.+)_x(\d+)y(\d+)\.([a-zA-Z0-9]+)$')
    
    file_data = []
    extensions = set()
    rows = set()
    cols = set()
    
    # Scan directory for matching files
    dir_path = Path(directory)
    for file in dir_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                prefix, x_val, y_val, ext = match.groups()
                x_int = int(x_val)
                y_int = int(y_val)
                
                file_data.append((y_int, x_int, file.name))
                extensions.add(ext)
                rows.add(y_int)
                cols.add(x_int)
    
    if not file_data:
        raise ValueError(f"No files matching pattern found in {directory}")
    
    # Check that all files have the same extension
    if len(extensions) > 1:
        raise ValueError(f"Multiple extensions found: {extensions}")
    
    extension = extensions.pop()
    
    # Create mappings from actual values to indices (0-based)
    sorted_rows = sorted(rows)
    sorted_cols = sorted(cols)
    row_to_idx = {row_val: idx for idx, row_val in enumerate(sorted_rows)}
    col_to_idx = {col_val: idx for idx, col_val in enumerate(sorted_cols)}
    
    # Create a set of found positions for validation
    found_positions = {(row_val, col_val) for row_val, col_val, _ in file_data}
    
    # Check for missing files
    missing_files = []
    for row_val in sorted_rows:
        for col_val in sorted_cols:
            if (row_val, col_val) not in found_positions:
                row_idx = row_to_idx[row_val]
                col_idx = col_to_idx[col_val]
                missing_files.append((row_idx, col_idx, row_val, col_val))
    
    # Print warnings for missing files
    if missing_files:
        print(f"⚠️  WARNING: {len(missing_files)} file(s) missing from grid:")
        for row_idx, col_idx, row_val, col_val in missing_files:
            print(f"   Missing at grid position [{row_idx}, {col_idx}] (x={col_val}, y={row_val})")
    
    # Build final file list with indices first
    file_list = []
    for row_val, col_val, filename in file_data:
        row_idx = row_to_idx[row_val]
        col_idx = col_to_idx[col_val]
        file_list.append((row_idx, col_idx, row_val, col_val, filename))
    
    # Sort by row index, then column index
    file_list.sort(key=lambda item: (item[0], item[1]))
    
    nrows = len(sorted_rows)
    ncols = len(sorted_cols)
    
    # Try to infer image information if the files are images
    image_info = None
    try:
        image_info = infer_image_info(directory)
        print(f"Image info: type={image_info[0]}, size={image_info[1]} (height×width)")
    except ValueError:
        # Not an image directory or couldn't read images
        pass
    
    return extension, nrows, ncols, file_list, image_info


def infer_image_info(directory: str) -> Tuple[str, Tuple[int, int]]:
    """
    Infer the type and size of image files in a directory.
    
    This function examines the first few image files in the directory to determine
    their format and dimensions, assuming all images have the same size.
    
    Args:
        directory: Path to directory containing image files
        
    Returns:
        tuple: (image_type, image_size)
            - image_type: Common image format (e.g., 'PNG', 'JPEG', 'TIFF')
            - image_size: Tuple of (height, width) in pixels
            
    Raises:
        ValueError: If no valid image files are found or if images have different sizes
    """
    dir_path = Path(directory)
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
    
    image_files = []
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        raise ValueError(f"No image files found in {directory}")
    
    # Check the first few images to determine format and size
    image_types = set()
    image_sizes = set()
    
    # Examine up to 5 images to verify consistency
    for img_file in image_files[:5]:
        try:
            with Image.open(img_file) as img:
                image_types.add(img.format)
                image_sizes.add(img.size)  # PIL returns (width, height)
        except Exception as e:
            print(f"Warning: Could not read {img_file}: {e}")
            continue
    
    if not image_types:
        raise ValueError(f"Could not read any valid image files in {directory}")
    
    if len(image_types) > 1:
        print(f"Warning: Multiple image formats found: {image_types}. Using {image_types.pop()}")
    
    if len(image_sizes) > 1:
        raise ValueError(f"Inconsistent image sizes found: {image_sizes}")
    
    # Convert from (width, height) to (height, width) for consistency
    width, height = image_sizes.pop()
    image_type = image_types.pop()
    
    return image_type, (height, width)


def find_grid_directory(filename: str) -> str:
    """
    Find the first subdirectory containing files matching the pattern: prefix_x<INT>y<INT>.<EXT>
    
    This function searches only within the directory of the given file for subdirectories
    that contain a set of files conforming to the grid pattern expected by organize_grid_files().
    
    Args:
        filename: Path to a file (can be any file in or near the grid directory)
        
    Returns:
        str: Path to the first directory containing files matching the grid pattern
        
    Raises:
        ValueError: If no directory with matching files is found
    """
    # Pattern to match: anything followed by _x<digits>y<digits>.<extension>
    pattern = re.compile(r'^(.+)_x(\d+)y(\d+)\.([a-zA-Z0-9]+)$')
    
    # Start from the directory containing the given file
    file_path = Path(filename)
    search_dir = file_path.parent
    
    # Extract the expected directory name (filename without extension)
    expected_dir_name = file_path.stem
    
    # Check all subdirectories in the current search directory (no recursion)
    for item in search_dir.iterdir():
        if item.is_dir():
            # Check if this directory contains files matching the pattern
            matching_files = []
            extensions = set()
            
            for sub_file in item.iterdir():
                if sub_file.is_file():
                    match = pattern.match(sub_file.name)
                    if match:
                        matching_files.append(sub_file.name)
                        extensions.add(match.groups()[3])
            
            # We need at least 4 files with the same extension to consider it a valid grid
            if len(matching_files) >= 4 and len(extensions) == 1:
                # Check if directory name matches expected name
                if item.name != expected_dir_name:
                    print(f"⚠️  WARNING: Directory name '{item.name}' does not match expected '{expected_dir_name}'")
                
                print(f"Found grid directory: {item}")
                print(f"  Contains {len(matching_files)} files with extension: {extensions.pop()}")
                return str(item)
    
    raise ValueError(f"No directory with Kikuchi pattern files found in {search_dir}")


def read_ang_file(filename):
    """
    Read EBSD .ang file and extract header information and data
    according to the VERSION specified in the header.
    """
    # Initialize variables
    header = {}
    data_lines = []
    in_data = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if we've reached the data section
            if not line.startswith('#'):
                in_data = True
                data_lines.append(line)
            else:
                # Parse header information
                if 'VERSION' in line:
                    header['VERSION'] = int(line.split()[-1])
                elif 'x-star' in line:
                    header['x-star'] = float(line.split()[-1])
                elif 'y-star' in line:
                    header['y-star'] = float(line.split()[-1])
                elif 'z-star' in line:
                    header['z-star'] = float(line.split()[-1])
                elif 'SampleTiltAngle' in line:
                    header['SampleTiltAngle'] = float(line.split()[-1])
                elif 'CameraElevationAngle' in line:
                    header['CameraElevationAngle'] = float(line.split()[-1])
                elif 'CameraAzimuthalAngle' in line:
                    header['CameraAzimuthalAngle'] = float(line.split()[-1])
                elif 'WorkingDistance' in line:
                    header['WorkingDistance'] = float(line.split()[-1])
                elif 'XSTEP:' in line:
                    header['XSTEP'] = float(line.split()[-1])
                elif 'YSTEP:' in line:
                    header['YSTEP'] = float(line.split()[-1])
                elif 'NCOLS_ODD:' in line:
                    header['NCOLS_ODD'] = int(line.split()[-1])
                elif 'NCOLS_EVEN:' in line:
                    header['NCOLS_EVEN'] = int(line.split()[-1])
                elif 'NROWS:' in line:
                    header['NROWS'] = int(line.split()[-1])
                elif 'COLUMN_COUNT:' in line:
                    header['COLUMN_COUNT'] = int(line.split()[-1])
                elif 'COLUMN_HEADERS:' in line:
                    # Extract column headers
                    headers_str = line.split('COLUMN_HEADERS:')[-1].strip()
                    header['COLUMN_HEADERS'] = [h.strip() for h in headers_str.split(',')]
    
    return header, data_lines


def parse_ang_data(header, data_lines):
    """
    Parse data lines into numpy array according to VERSION format.
    """
    # Determine number of columns based on VERSION
    version = header.get('VERSION', 7)
    n_cols = header.get('COLUMN_COUNT', 13)
    
    # Parse data
    data = []
    for line in data_lines:
        values = line.split()
        if len(values) == n_cols:
            data.append([float(v) for v in values])
    
    data_array = np.array(data)
    
    # Create dictionary with column names
    col_headers = header.get('COLUMN_HEADERS', [])
    data_dict = {}
    for i, col_name in enumerate(col_headers):
        if i < data_array.shape[1]:
            data_dict[col_name] = data_array[:, i]
    
    return data_dict, data_array


def create_2d_maps(header, data_dict):
    """
    Reshape 1D data arrays into 2D maps based on grid dimensions.
    """
    nrows = header['NROWS']
    ncols = header['NCOLS_ODD']  # Assuming square grid
    
    maps_2d = {}
    for col_name, values in data_dict.items():
        # Reshape to 2D grid
        if len(values) == nrows * ncols:
            maps_2d[col_name] = values.reshape((nrows, ncols))
        else:
            print(f"Warning: {col_name} has {len(values)} values, expected {nrows*ncols}")
            # Pad or truncate as needed
            padded = np.zeros(nrows * ncols)
            padded[:len(values)] = values[:nrows*ncols]
            maps_2d[col_name] = padded.reshape((nrows, ncols))
    
    return maps_2d


def plot_maps(header, maps_2d, columns_to_plot=None):
    """
    Plot 2D maps for selected columns.
    """
    if columns_to_plot is None:
        # Default: plot some interesting columns
        columns_to_plot = ['IQ', 'CI', 'Fit', 'phi1', 'PHI', 'phi2']
    
    # Filter to only existing columns
    columns_to_plot = [col for col in columns_to_plot if col in maps_2d]
    
    n_plots = len(columns_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    xstep = header['XSTEP']
    ystep = header['YSTEP']
    ncols = header['NCOLS_ODD']
    nrows = header['NROWS']
    
    extent = [0, ncols * xstep, 0, nrows * ystep]
    
    for idx, col_name in enumerate(columns_to_plot[:6]):
        ax = axes[idx]
        data = maps_2d[col_name]
        
        im = ax.imshow(data, extent=extent, origin='lower', 
                      cmap='viridis', aspect='auto')
        ax.set_xlabel('X (microns)')
        ax.set_ylabel('Y (microns)')
        ax.set_title(f'{col_name}')
        plt.colorbar(im, ax=ax, label=col_name)
    
    # Hide unused subplots
    for idx in range(len(columns_to_plot), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('ebsd_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# Main execution
def main(filename):
    """
    Main function to read and visualize EBSD data.
    """
    print("Reading .ang file...")
    header, data_lines = read_ang_file(filename)
    
    print(f"\nHeader Information:")
    print(f"  VERSION: {header.get('VERSION', 'N/A')}")
    print(f"  Grid size: {header.get('NROWS', 'N/A')} rows × {header.get('NCOLS_ODD', 'N/A')} cols")
    print(f"  Step size: {header.get('XSTEP', 'N/A')} × {header.get('YSTEP', 'N/A')} microns")
    print(f"  Number of columns: {header.get('COLUMN_COUNT', 'N/A')}")
    print(f"  Column headers: {header.get('COLUMN_HEADERS', 'N/A')}")
    
    print(f"\nParsing data...")
    data_dict, data_array = parse_ang_data(header, data_lines)
    print(f"  Data shape: {data_array.shape}")
    print(f"  Total data points: {len(data_lines)}")
    
    '''
    print(f"\nCreating 2D maps...")
    maps_2d = create_2d_maps(header, data_dict)
    
    print(f"\nAvailable maps:")
    for col_name in maps_2d.keys():
        print(f"  - {col_name}: shape {maps_2d[col_name].shape}, "
              f"range [{maps_2d[col_name].min():.3f}, {maps_2d[col_name].max():.3f}]")
    
    print(f"\nPlotting maps...")
    fig = plot_maps(header, maps_2d)
    
    print("\nDone! Maps saved as 'ebsd_maps.png'")
    '''
    
    return header, data_dict  #, maps_2d

# Example usage:
if __name__ == "__main__":
    filename = "./data/Mo CC-EBSD-ECCI_area1/Mo CC-EBSD-ECCI_area1.ang"
    header, data_dict = main(filename)
    print(header)
    print(data_dict)
