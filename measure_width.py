import numpy as np
from PIL import Image
import os
import glob

# Configuration
MAPS_DIR = "maps_gradient"
METERS_PER_PIXEL = 0.04

def measure_effective_width(image_path, fixed_threshold=None):
    """
    Measures the effective width of the path from a generated gradient image.
    
    Args:
        image_path (str): Path to the image file.
        fixed_threshold (int, optional): If provided, uses this specific pixel value (0-255)
                                         as the cutoff. Otherwise, uses Half-Maximum (FWHM).
    """
    try:
        # Load image and convert to grayscale
        img = Image.open(image_path).convert('L')
        data = np.array(img)
        height, width = data.shape
        
        # Take a vertical slice from the middle of the image
        mid_x = width // 2
        col_slice = data[:, mid_x]
        
        min_val = np.min(col_slice)
        max_val = np.max(col_slice)
        
        # Determine the threshold
        if fixed_threshold is not None:
            threshold = fixed_threshold
            method_name = f"Fixed({fixed_threshold})"
        else:
            # Default: Half-Width at Half-Maximum (FWHM)
            dynamic_range = max_val - min_val
            if dynamic_range < 10:
                return 0, 0, "Flat/Low Contrast"
            threshold = min_val + (dynamic_range / 2.0)
            method_name = "Half-Max"
        
        # Find pixels that are part of the "path" (assuming path is darker)
        # We select all pixels that are darker (lower value) than the threshold
        path_indices = np.where(col_slice <= threshold)[0]
            
        if len(path_indices) == 0:
             return 0, 0, "No distinct path found"
             
        # Calculate width as the span between the first and last index
        # This handles gaps slightly better than just len(), though for a single convex path they are the same
        pixel_width = path_indices[-1] - path_indices[0] + 1
        
        # Convert to meters
        meter_width = pixel_width * METERS_PER_PIXEL
        
        return pixel_width, meter_width, method_name, min_val, max_val, threshold

    except Exception as e:
        return 0, 0, f"Error: {e}", 0, 0, 0

def main():
    #Files we are interested in
    target_files = [
        "path_example_20.0_sine_curve_width0.2_freq2.png",
        "path_example_20.0_sine_curve_width0.4_freq2.png",
        "path_example_20.0_sine_curve_width10.0_freq2.png"
    ]

    print(f"{'Filename':<50} | {'Pixels':<6} | {'Meters':<8} | {'Method':<10} | {'Min/Max':<10}")
    print("-" * 105)

    for filename in target_files:
        filepath = os.path.join(MAPS_DIR, filename)
        if os.path.exists(filepath):
            # Run with Half-Max (Standard)
            pixels, meters, method, min_v, max_v, thresh = measure_effective_width(filepath)
            print(f"{filename:<50} | {pixels:<6} | {meters:<8.4f} m | {method:<10} | {min_v}/{max_v} (Thresh: {thresh:.1f})")

            # Run with Fixed Threshold = 150
            pixels_150, meters_150, method_150, _, _, _ = measure_effective_width(filepath, fixed_threshold=151)
            print(f"{'':<50} | {pixels_150:<6} | {meters_150:<8.4f} m | {method_150:<10} | Thresh: 150")

            # OPTIONAL: ASCII Visualization for ALL files
            print(f"\n   [DEBUG] Pixel values for {filename} (center slice):")
            img = Image.open(filepath).convert('L')
            data = np.array(img)
            mid_col = data[:, data.shape[1]//2]
            
            # For the thin one, show the small window
            if "thick0.90" in filename:
                 start, end = 45, 55
                 for r in range(start, end):
                     val = mid_col[r]
                     # Check against Fixed(150) threshold
                     marker = "<-- PATH" if val <= 150 else ""
                     print(f"   Row {r}: {val:3d} {marker}")
            
            # For the wide ones
            else:
                 for r in range(0, 100, 10):
                     val = mid_col[r]
                     marker = "<-- PATH" if val <= 150 else "        "
                     print(f"   Row {r:02d}: {val:3d} {marker}")
            
            print("-" * 105)

        else:
            print(f"{filename:<50} | File not found")

if __name__ == "__main__":
    main()

