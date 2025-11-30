import numpy as np
from PIL import Image
from noise import pnoise2
import os

# --- Configuration ---
# Dimensions calculated from: world_size / step_size (step_size = 0.04 from dm_ds_v2.py)
# WORLD_SIZE_X = 20.0m -> 20.0 / 0.04 = 500 pixels
# WORLD_SIZE_Y = 4.0m -> 4.0 / 0.04 = 100 pixels
WIDTH = 500  # Updated for 20m world (was 163 for 6.5m)
HEIGHT = 100  # Unchanged (still 4.0m)

# Auto-detect which computer we're on by checking which base path exists
_MAC_BASE = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient"
_LINUX_BASE = "/home/ksb8405/Documents/PyBullet/gym-pybullet-drones-3DAE/maps_gradient"

# Determine base path based on which directory exists
if os.path.exists(os.path.dirname(_MAC_BASE)):
    OUTPUT_DIR = _MAC_BASE
elif os.path.exists(os.path.dirname(_LINUX_BASE)):
    OUTPUT_DIR = _LINUX_BASE
else:
    # Fallback: use relative path from current working directory
    OUTPUT_DIR = os.path.join(os.getcwd(), "maps_gradient")
    print(f"⚠️  WARNING: Neither Mac nor Linux path found, using cwd: {OUTPUT_DIR}")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_and_save(gradient, filename, invert_colors=False, max_val=255):
    """
    Normalize a 2D numpy array to 0-max_val and save as a grayscale PNG.
    
    Args:
        gradient: 2D numpy array of values
        filename: Output filename
        invert_colors: If True, bright becomes dark (0)
        max_val: The maximum brightness value (0-255). 
                 If 150, the brightest pixel will be 150 (dark gray).
    """
    # Normalize to 0-1 range
    gradient_min = np.min(gradient)
    gradient_max = np.max(gradient)
    
    # DEBUG PRINT
    print(f"  [Debug] File: {filename} | Min: {gradient_min:.4f} | Max: {gradient_max:.4f} | Target MaxVal: {max_val}")
    
    if gradient_max - gradient_min > 0:
        normalized = (gradient - gradient_min) / (gradient_max - gradient_min)
    else:
        normalized = np.zeros_like(gradient)

    # Invert colors if requested
    if invert_colors:
        normalized = 1.0 - normalized

    # Scale to 0-max_val and convert to uint8
    image_data = np.uint8(normalized * max_val)

    # Create and save image
    img = Image.fromarray(image_data, 'L')
    output_path = os.path.join(OUTPUT_DIR, filename)
    img.save(output_path)
    print(f"✅ Gradient saved to: {output_path}")


def thicken_path(path_function, width, height, thickness=0.05, invert=False):
    """
    Thickens any path/curve by creating an exponential falloff around it.
    
    Args:
        path_function: A function that takes x coordinates (0 to width) and returns y coordinates
        width: Width of the output image
        height: Height of the output image
        thickness: Controls the thickness of the path (smaller = thicker, typical range: 0.01-0.1)
        invert: If True, the path is dark (0.0) and surroundings are bright (1.0)
                If False, the path is bright (1.0) and surroundings are dark (0.0)
    
    Returns:
        A 2D numpy array with values between 0 and 1
    """
    # print(f"Thickening path with thickness parameter: {thickness}, invert: {invert}")
    
    # Create coordinate grids
    y, x = np.mgrid[0:height, 0:width]
    
    # Calculate the path y-coordinates for each x position
    path_y = path_function(x)
    
    # Calculate the distance from each point to the path
    distance = np.abs(y - path_y)
    
    # Apply exponential falloff to create thickness
    thickened = np.exp(-thickness * distance)
    
    # Invert if requested (path becomes dark instead of bright)
    if invert:
        thickened = 1.0 - thickened
    
    return thickened


def get_thickness_for_width(target_width_meters, threshold=150):
    """
    Calculates the thickness parameter 'k' needed to achieve a specific effective width.
    """
    meters_per_pixel = 0.04
    width_pixels = target_width_meters / meters_per_pixel
    d_pixels = width_pixels / 2.0
    
    # Target fraction of brightness at the edge of the width
    # If max brightness is 150, then the threshold IS the max brightness.
    # This means we want the gradient to reach the max brightness at the edge of the width.
    # But wait, usually the gradient continues past the "effective width".
    # If we clamp the output image to 0-150, then "effective width" defined as <= 150 
    # will be the ENTIRE IMAGE again.
    
    # Instead, assume the user wants the "slope" to be such that the value hits 150 
    # at the specified width, assuming a theoretical max of 255.
    # BUT then we render it with max_val=150? That cuts off the data.
    
    # Let's stick to the standard calculation assuming 0-255 range for the shape,
    # and then we render it darker.
    
    target_fraction = threshold / 255.0
    
    term = 1.0 - target_fraction
    k = -np.log(term) / d_pixels
    
    return k

def generate_linear_gradient(width, height):
    """Generates a simple linear gradient from dark (left) to bright (right)."""
    print("Generating linear gradient...")
    gradient = np.linspace(0, 1, width)
    gradient = np.tile(gradient, (height, 1))
    return gradient


def generate_parabolic_funnel(width, height):
    """Generates a gradient with a left-to-right flow and a vertical parabolic funnel."""
    print("Generating parabolic funnel gradient...")
    y, x = np.mgrid[-1:1:height*1j, 0:1:width*1j]
    a = 1.0 
    b = 2.0
    gradient = a * x + b * (y**2)
    return gradient


def generate_noisy_flow(width, height):
    """Generates a gradient with a left-to-right flow combined with Perlin noise."""
    print("Generating noisy flow gradient...")
    y, x = np.mgrid[0:height, 0:width]
    scale = 150.0
    octaves = 4
    persistence = 0.6
    lacunarity = 2.0
    perlin_grid = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            perlin_grid[i][j] = pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=42)
    linear_gradient = np.linspace(0, 1, width)
    linear_grid = np.tile(linear_gradient, (height, 1))
    gradient = 0.6 * linear_grid + 0.4 * perlin_grid
    return gradient


def generate_sine_wave(width, height):
    """Generates a gradient with a sine wave path."""
    print("Generating sine wave gradient...")
    amplitude = height / 4
    frequency = 2 * np.pi / width
    def sine_path(x):
        return amplitude * np.sin(frequency * x) + height / 2
    thickness = 0.05
    sine_falloff = thicken_path(sine_path, width, height, thickness)
    linear_ramp = np.linspace(0.5, 1.0, width)
    linear_grid = np.tile(linear_ramp, (height, 1))
    gradient = sine_falloff * linear_grid
    return gradient


def generate_noisy_sine_wave(width, height):
    """Generates a noisy, ramped sine wave gradient."""
    print("Generating noisy sine wave gradient...")
    y, x = np.mgrid[0:height, 0:width]
    amplitude = height / 4
    frequency = 2 * np.pi / width
    def sine_path(x):
        return amplitude * np.sin(frequency * x) + height / 2
    thickness = 0.05
    sine_falloff = thicken_path(sine_path, width, height, thickness)
    linear_ramp = np.linspace(0.5, 1.0, width)
    linear_grid = np.tile(linear_ramp, (height, 1))
    scale = 100.0
    octaves = 5
    persistence = 0.5
    lacunarity = 2.0
    perlin_grid = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            perlin_grid[i][j] = pnoise2(i / scale, j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        base=42)
    p_min, p_max = np.min(perlin_grid), np.max(perlin_grid)
    centered_perlin = 2 * (perlin_grid - p_min) / (p_max - p_min) - 1
    noise_strength = 0.5
    noisy_ramp = linear_grid * (1 + centered_perlin * noise_strength)
    gradient = sine_falloff * noisy_ramp
    return gradient


def generate_ramped_sine_with_banks(width, height):
    """Generates a ramped sine wave with a vertical gradient on its banks."""
    print("Generating ramped sine wave with banks...")
    y, x = np.mgrid[0:height, 0:width]
    amplitude = height / 4
    frequency = 2 * np.pi / width
    def sine_path(x):
        return amplitude * np.sin(frequency * x) + height / 2
    thickness = 0.05
    sine_falloff = thicken_path(sine_path, width, height, thickness)
    sine_wave = sine_path(x)
    distance_from_sine = np.abs(y - sine_wave)
    linear_ramp = np.linspace(0.5, 1.0, width)
    linear_grid = np.tile(linear_ramp, (height, 1))
    ramped_sine_path = sine_falloff * linear_grid
    bank_strength = 0.005
    banks_gradient = bank_strength * distance_from_sine
    path_mask = 1.0 - sine_falloff
    combined_gradient = ramped_sine_path + (path_mask * banks_gradient)
    return combined_gradient


def generate_custom_path_examples(width, height):
    """Demonstrates how to use thicken_path with various mathematical functions."""
    print("Generating custom path examples...")
    
    def parabola_path(x):
        return 0.001 * (x - width / 2) ** 2 + height / 4
    
    def exponential_path(x):
        normalized_x = x / width
        return height * (1 - np.exp(-3 * normalized_x))
    
    def zigzag_path(x):
        period = width / 5
        return height / 2 + (height / 4) * np.abs((x % period) / period * 2 - 1)
    
    def circular_arc_path(x):
        center_x = width / 2
        radius = width / 2
        x_offset = x - center_x
        valid = np.abs(x_offset) <= radius
        result = np.full_like(x, height / 2, dtype=float)
        result[valid] = height / 2 - np.sqrt(radius**2 - x_offset[valid]**2) * (height / (2 * radius))
        return result
    
    def sine_curve_path(x, freq):
        amplitude = height / 4
        frequency = freq * np.pi / width
        return amplitude * np.sin(frequency * x) + height / 2
    
    examples = {
        'parabola': thicken_path(parabola_path, width, height, thickness=0.03, invert=True),
        'exponential': thicken_path(exponential_path, width, height, thickness=0.001, invert=True),
        'zigzag': thicken_path(zigzag_path, width, height, thickness=0.05, invert=True),
        'circular_arc': thicken_path(circular_arc_path, width, height, thickness=0.06, invert=True),
    }
    
    # ANTS 2026 Experiment
    target_widths = {
        'width0.2': 0.2,  # 20cm wide
        'width0.4': 0.4,  # 40cm wide
        'width10.0': 10.0,  # Effectively infinite
    }
    
    # Calculate k assuming a standard 0-255 gradient where threshold=150
    thickness_levels = {}
    print("\nCalculating thickness parameters for target widths (Threshold=150):")
    for name, w in target_widths.items():
        k = get_thickness_for_width(w, threshold=150)
        thickness_levels[name] = k
        print(f"  {name}: {w}m -> k={k:.4f}")

    frequency_levels = {
        'freq2': 2,
        'freq4': 4,
        'freq8': 8,
    }
    
    print(f"\nGenerating 3x3 grid of sine curves:")
    for width_name, thickness_val in thickness_levels.items():
        for freq_name, freq_val in frequency_levels.items():
            map_name = f'sine_curve_{width_name}_{freq_name}'
            print(f"  Creating: {map_name} (thickness={thickness_val:.4f}, freq={freq_val})")
            sine_path_func = lambda x, f=freq_val: sine_curve_path(x, f)
            
            # Create the thickened path
            gradient = thicken_path(sine_path_func, width, height, thickness=thickness_val, invert=True)
            
            # Store in examples
            examples[map_name] = gradient
    
    exponential_thickened = thicken_path(exponential_path, width, height, thickness=0.001, invert=True)
    linear_ramp = np.linspace(0.0, 1.0, width)
    linear_grid = np.tile(linear_ramp, (height, 1))
    exponential_with_gradient = exponential_thickened * (1 - linear_grid * 0.5)
    examples['exponential_gradient'] = exponential_with_gradient
    
    return examples


if __name__ == "__main__":
    print("--- Generating Gradient Maps ---")

    # Standard gradients (kept at full range 0-255)
    linear_gradient = generate_linear_gradient(WIDTH, HEIGHT)
    normalize_and_save(linear_gradient, "linear_gradient.png")
    normalize_and_save(linear_gradient, "linear_gradient_inverted.png", invert_colors=True)

    parabolic_gradient = generate_parabolic_funnel(WIDTH, HEIGHT)
    normalize_and_save(parabolic_gradient, "parabolic_funnel.png")
    normalize_and_save(parabolic_gradient, "parabolic_funnel_inverted.png", invert_colors=True)

    noisy_gradient = generate_noisy_flow(WIDTH, HEIGHT)
    normalize_and_save(noisy_gradient, "noisy_flow.png")

    sine_wave_gradient = generate_sine_wave(WIDTH, HEIGHT)
    normalize_and_save(sine_wave_gradient, "sine_wave_gradient_ramped.png", invert_colors=False)

    noisy_sine_wave_gradient = generate_noisy_sine_wave(WIDTH, HEIGHT)
    normalize_and_save(noisy_sine_wave_gradient, "sine_wave_noisy.png", invert_colors=True)

    ramped_sine_with_banks = generate_ramped_sine_with_banks(WIDTH, HEIGHT)
    normalize_and_save(ramped_sine_with_banks, "sine_wave_ramped_with_banks.png", invert_colors=True)
    
    # Custom examples (using reduced max_val for the sine curves)
    print("\n--- Generating Custom Path Examples ---")
    custom_examples = generate_custom_path_examples(WIDTH, HEIGHT)
    for name, gradient in custom_examples.items():
        
        # Apply MAX_BRIGHTNESS = 150 for the sine curve experiment files
        # This ensures the background is dark gray, not white.
        if "sine_curve" in name:
            normalize_and_save(gradient, f"path_example_{WIDTH*0.04}_{name}.png", 
                             invert_colors=False, max_val=75)
        else:
            # Keep other examples normal
            normalize_and_save(gradient, f"path_example_{WIDTH*0.04}_{name}.png", 
                             invert_colors=False, max_val=255)

    print("\nAll gradients generated successfully!")
