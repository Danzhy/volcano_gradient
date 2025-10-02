import numpy as np
from PIL import Image
from noise import pnoise2
import os

# --- Configuration ---
# Dimensions calculated from: world_size / step_size (step_size = 0.04 from dm_ds_v2.py)
# WORLD_SIZE_X = 6.5m -> 6.5 / 0.04 = 162.5 ≈ 163 pixels
# WORLD_SIZE_Y = 4.0m -> 4.0 / 0.04 = 100 pixels
WIDTH = 163  # Changed from 650 to match coordinate mapping
HEIGHT = 100  # Changed from 400 to match coordinate mapping
OUTPUT_DIR = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_and_save(gradient, filename, invert_colors=False):
    """Normalize a 2D numpy array to 0-255 and save as a grayscale PNG."""
    # Normalize to 0-1 range
    gradient_min = np.min(gradient)
    gradient_max = np.max(gradient)
    if gradient_max - gradient_min > 0:
        normalized = (gradient - gradient_min) / (gradient_max - gradient_min)
    else:
        normalized = np.zeros_like(gradient)

    # Invert colors if requested
    if invert_colors:
        normalized = 1.0 - normalized

    # Scale to 0-255 and convert to uint8
    image_data = np.uint8(normalized * 255)

    # Create and save image
    img = Image.fromarray(image_data, 'L')
    output_path = os.path.join(OUTPUT_DIR, filename)
    img.save(output_path)
    print(f"✅ Gradient saved to: {output_path}")


def generate_linear_gradient(width, height):
    """Generates a simple linear gradient from dark (left) to bright (right)."""
    print("Generating linear gradient...")
    # Create a linear gradient from 0 to 1 across the width
    gradient = np.linspace(0, 1, width)
    # Tile it vertically to fill the entire height
    gradient = np.tile(gradient, (height, 1))
    return gradient


def generate_parabolic_funnel(width, height):
    """Generates a gradient with a left-to-right flow and a vertical parabolic funnel."""
    print("Generating parabolic funnel gradient...")
    y, x = np.mgrid[-1:1:height*1j, 0:1:width*1j]

    # Coefficients for the gradient components
    a = 1.0  # Strength of the left-to-right flow
    b = 2.0  # Strength of the funneling effect

    # Parabolic funnel equation: G(x, y) = a*x + b*y^2
    gradient = a * x + b * (y**2)

    return gradient


def generate_noisy_flow(width, height):
    """Generates a gradient with a left-to-right flow combined with Perlin noise."""
    print("Generating noisy flow gradient...")
    y, x = np.mgrid[0:height, 0:width]

    # Noise parameters
    scale = 150.0  # Controls the "zoom" level of the noise
    octaves = 4
    persistence = 0.6
    lacunarity = 2.0

    # Generate Perlin noise
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

    # Create a linear gradient from left to right
    linear_gradient = np.linspace(0, 1, width)
    linear_grid = np.tile(linear_gradient, (height, 1))

    # Combine linear gradient and Perlin noise
    # Adjust weights to control the influence of each component
    gradient = 0.6 * linear_grid + 0.4 * perlin_grid

    return gradient


def generate_sine_wave(width, height):
    """Generates a gradient with a sine wave path that is thicker and has an increasing gradient."""
    print("Generating sine wave gradient...")
    y, x = np.mgrid[0:height, 0:width]

    # Sine wave parameters
    amplitude = height / 4  # Controls the height of the wave
    frequency = 2 * np.pi / width  # Controls the number of cycles

    # Sine wave equation
    sine_wave = amplitude * np.sin(frequency * x) + height / 2

    # Calculate the distance from each point to the sine wave
    distance = np.abs(y - sine_wave)

    # 1. Create a "thick" path using exponential falloff
    thickness = 0.05  # Smaller value = thicker path
    sine_falloff = np.exp(-thickness * distance)

    # 2. Create a linear gradient to make the path brighter from left to right
    linear_ramp = np.linspace(0.5, 1.0, width)  # Ramps from gray to white
    linear_grid = np.tile(linear_ramp, (height, 1))

    # 3. Combine them: path brightness is modulated by the linear ramp
    gradient = sine_falloff * linear_grid

    return gradient


def generate_noisy_sine_wave(width, height):
    """Generates a noisy, ramped sine wave gradient, like a lava river."""
    print("Generating noisy sine wave gradient...")
    y, x = np.mgrid[0:height, 0:width]

    # --- 1. Generate the sine wave path shape ---
    amplitude = height / 4
    frequency = 2 * np.pi / width
    sine_wave = amplitude * np.sin(frequency * x) + height / 2
    distance = np.abs(y - sine_wave)
    thickness = 0.05
    sine_falloff = np.exp(-thickness * distance)

    # --- 2. Generate the underlying texture (ramped noise) ---
    # Linear ramp from left to right
    linear_ramp = np.linspace(0.5, 1.0, width)
    linear_grid = np.tile(linear_ramp, (height, 1))

    # Perlin noise
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
    # Normalize noise to be in [-1, 1] range
    p_min, p_max = np.min(perlin_grid), np.max(perlin_grid)
    centered_perlin = 2 * (perlin_grid - p_min) / (p_max - p_min) - 1

    # Combine ramp and noise
    noise_strength = 0.5
    noisy_ramp = linear_grid * (1 + centered_perlin * noise_strength)

    # --- 3. Apply the path shape to the texture ---
    gradient = sine_falloff * noisy_ramp

    return gradient


def generate_ramped_sine_with_banks(width, height):
    """Generates a ramped sine wave with a vertical gradient on its banks."""
    print("Generating ramped sine wave with banks...")
    y, x = np.mgrid[0:height, 0:width]

    # --- 1. Define the sine wave path ---
    amplitude = height / 4
    frequency = 2 * np.pi / width
    sine_wave = amplitude * np.sin(frequency * x) + height / 2
    distance_from_sine = np.abs(y - sine_wave)

    # --- 2. Create the ramped sine path component ---
    thickness = 0.05
    sine_falloff = np.exp(-thickness * distance_from_sine)
    linear_ramp = np.linspace(0.5, 1.0, width)
    linear_grid = np.tile(linear_ramp, (height, 1))
    ramped_sine_path = sine_falloff * linear_grid

    # --- 3. Create the vertical gradient "banks" ---
    bank_strength = 0.005  # Controls how quickly the banks get darker
    banks_gradient = bank_strength * distance_from_sine

    # --- 4. Combine the path and the banks ---
    # We want the banks to be dark where the path is not, so we invert the path's mask
    path_mask = 1.0 - sine_falloff
    combined_gradient = ramped_sine_path + (path_mask * banks_gradient)

    return combined_gradient


if __name__ == "__main__":
    print("--- Generating Gradient Maps ---")

    # Generate and save the linear gradient
    linear_gradient = generate_linear_gradient(WIDTH, HEIGHT)
    normalize_and_save(linear_gradient, "linear_gradient.png")
    normalize_and_save(linear_gradient, "linear_gradient_inverted.png", invert_colors=True)

    # Generate and save the parabolic funnel
    parabolic_gradient = generate_parabolic_funnel(WIDTH, HEIGHT)
    normalize_and_save(parabolic_gradient, "parabolic_funnel.png")
    normalize_and_save(parabolic_gradient, "parabolic_funnel_inverted.png", invert_colors=True)

    # Generate and save the noisy flow
    noisy_gradient = generate_noisy_flow(WIDTH, HEIGHT)
    normalize_and_save(noisy_gradient, "noisy_flow.png")

    # Generate and save the ramped sine wave gradient
    sine_wave_gradient = generate_sine_wave(WIDTH, HEIGHT)
    normalize_and_save(sine_wave_gradient, "sine_wave_gradient_ramped.png", invert_colors=False)

    # Generate and save the noisy sine wave gradient
    noisy_sine_wave_gradient = generate_noisy_sine_wave(WIDTH, HEIGHT)
    normalize_and_save(noisy_sine_wave_gradient, "sine_wave_noisy.png", invert_colors=True)

    # Generate and save the ramped sine wave with banks
    ramped_sine_with_banks = generate_ramped_sine_with_banks(WIDTH, HEIGHT)
    normalize_and_save(ramped_sine_with_banks, "sine_wave_ramped_with_banks.png", invert_colors=True)

    print("\nAll gradients generated successfully!")
