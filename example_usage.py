"""
Example usage of Force Inference Python implementation

This script demonstrates how to use the ForceInference class
to analyze epithelial cell images.
"""

from force_inference import ForceInference
import os


def example_basic():
    """
    Basic example: Run force inference on a single image with border.
    """
    print("="*70)
    print("EXAMPLE 1: Basic Usage with Border")
    print("="*70)

    # Path to your image
    image_path = "for ReadMe/image.tif"

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please update the path to point to your image file.")
        return

    # Create ForceInference object
    # has_border=True means the image has filled borders
    fi = ForceInference(image_path, has_border=True)

    # Run the analysis
    result = fi.run()

    # Results are saved as:
    # - force_inference_results.png (tension and pressure maps)
    # - log_likelihood.png (optimization curve)

    print(f"\nResults:")
    print(f"  - Total parameters computed: {len(result)}")
    print(f"  - Output files generated:")
    print(f"    * force_inference_results.png")
    print(f"    * log_likelihood.png")


def example_no_border():
    """
    Example: Run force inference on an image without border.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Usage without Border")
    print("="*70)

    # Path to your image without border
    image_path = "my_image_no_border.tif"

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Note: Example image '{image_path}' not found.")
        print("This example shows how to analyze images without borders.")
        print("Set has_border=False to use morphological component segmentation.")
        return

    # Create ForceInference object
    # has_border=False means the image does not have filled borders
    fi = ForceInference(image_path, has_border=False)

    # Run the analysis
    result = fi.run()

    print(f"\nResults computed: {len(result)} parameters")


def example_custom_analysis():
    """
    Example: Access intermediate results for custom analysis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Analysis with Intermediate Results")
    print("="*70)

    image_path = "for ReadMe/image.tif"

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Create ForceInference object
    fi = ForceInference(image_path, has_border=True)

    # Run the analysis
    result = fi.run()

    # Access intermediate results
    print(f"\nIntermediate Results:")
    print(f"  - Image shape: {fi.image.shape}")
    print(f"  - Segmentation shape: {fi.segmentation.shape}")
    print(f"  - Number of cells: {fi.max_cell_labels}")

    # Extract tension and pressure values separately
    # (Note: In the actual run() method, these would need to be returned/stored)
    print(f"\nFinal solution vector contains:")
    print(f"  - Tension values for edges")
    print(f"  - Pressure values for cells")


def main():
    """
    Run all examples.
    """
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "FORCE INFERENCE - EXAMPLE USAGE" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    print()

    # Run Example 1: Basic usage
    try:
        example_basic()
    except Exception as e:
        print(f"Example 1 failed with error: {e}")

    # Run Example 2: No border
    try:
        example_no_border()
    except Exception as e:
        print(f"Example 2 skipped or failed: {e}")

    # Run Example 3: Custom analysis
    try:
        example_custom_analysis()
    except Exception as e:
        print(f"Example 3 failed with error: {e}")

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nFor more information, see PYTHON_README.md")


if __name__ == "__main__":
    main()
