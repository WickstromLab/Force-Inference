# Force-Inference for Epithelial Cells - Python Implementation

**Author:** Converted from Mathematica by Claude
**Original Author:** Ali Hashmi

This is a Python implementation of the **Bayesian Force Inference** technique by Ishihara and Sugimura (*Journal of Theoretical Biology*, 2012). The script infers tension for edges between epithelial cells and the pressure within the cells from a binarized image of an epithelia.

## Original Mathematica Implementation

For the original Mathematica implementation, see the [Wolfram Community post](https://community.wolfram.com/groups/-/m/t/1571507).

## Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-image
- Matplotlib
- Pillow

## Installation

1. Clone this repository:
```bash
git clone https://github.com/alihashmiii/Force-Inference.git
cd Force-Inference
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda install numpy scipy scikit-image matplotlib pillow
```

## Usage

### Command Line

Run the force inference analysis from the command line:

```bash
python force_inference.py <image_path> [has_border]
```

**Parameters:**
- `image_path`: Path to the binarized image file (e.g., `.tif`, `.png`)
- `has_border`: Optional boolean (`True` or `False`). Default is `True`.
  - `True`: Image has filled borders (uses watershed segmentation)
  - `False`: Image has no borders (uses morphological components)

**Example:**
```bash
# With border (default)
python force_inference.py "for ReadMe/image.tif" True

# Without border
python force_inference.py my_image.png False
```

### Python API

You can also use the `ForceInference` class in your own Python scripts:

```python
from force_inference import ForceInference

# Initialize with image path
fi = ForceInference("path/to/image.tif", has_border=True)

# Run analysis
result = fi.run()

# result contains the computed tensions and pressures
print(f"Computed {len(result)} parameters")
```

## Output

The script generates two output files:

1. **`force_inference_results.png`**: Visualization showing:
   - **Tension Map** (left): Edge tensions color-coded by magnitude
   - **Pressure Map** (right): Cell pressures color-coded by magnitude

2. **`log_likelihood.png`**: Plot of log-likelihood vs. regularization parameter μ

## How It Works

The force inference pipeline consists of several steps:

### 1. Image Segmentation
- Converts the input image to binary format
- Segments cells using either:
  - **Watershed segmentation** (when borders are present)
  - **Morphological components** (when borders are absent)

### 2. Vertex Detection
- Identifies vertices where 3 or more cell edges meet
- Associates each vertex with its neighboring cells
- Merges nearby vertices within 3 pixels

### 3. Edge Detection
- Identifies edges connecting vertices
- Creates edge-vertex connectivity

### 4. Matrix Formation
- **Tension coefficients**: Unit vectors along each edge
- **Pressure coefficients**: Perpendicular vectors for each cell
- Forms sparse matrices for efficient computation

### 5. Maximum Likelihood Optimization
- Searches over regularization parameter μ
- Uses QR decomposition for numerical stability
- Maximizes log-likelihood function
- Solves for optimal tension and pressure values

### 6. Visualization
- Generates color-coded maps showing:
  - Tension distribution along edges
  - Pressure distribution within cells

## Important Notes

- **Image Format**: The input should be a binarized image where cell boundaries are white (1) and cell interiors are black (0).
- **Border Pixels**: For best results with the test image (`image.tif`), ensure border pixels are filled.
- **Output**: Results are saved as PNG images in the current directory.

## Differences from Mathematica Version

This Python implementation maintains the core algorithm while adapting to Python's ecosystem:

- Uses **scikit-image** for morphological operations instead of Mathematica's built-in functions
- Uses **SciPy sparse matrices** for efficient matrix operations
- Uses **QR decomposition** from SciPy's linear algebra module
- Visualization with **Matplotlib** instead of Mathematica's Graphics
- Object-oriented design with the `ForceInference` class

## Algorithm Reference

> **Ishihara, S., & Sugimura, K.** (2012). *Bayesian inference of force dynamics during morphogenesis.* Journal of Theoretical Biology, 313, 201-211.

For more details, see also:
- [bioRxiv preprint by Lenne Lab (2018)](https://www.biorxiv.org/content/early/2018/11/23/475012)

## Example Output

The algorithm produces:
- **Tension values**: One value per edge (typically 10-100+ edges)
- **Pressure values**: One value per cell (typically 10-50+ cells)
- **Visualization**: Color-coded maps showing force distribution

## Troubleshooting

**Issue: "No vertices found"**
- Check that your image is properly binarized
- Ensure cell boundaries are continuous
- Try adjusting the `has_border` parameter

**Issue: "Matrix dimension mismatch"**
- This usually indicates incomplete cell segmentation
- Check that all cells are properly closed
- Verify the image has adequate resolution

**Issue: Poor visualization quality**
- Increase image DPI in the plot saving commands
- Adjust color map ranges if needed
- Check that vertices are correctly detected

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This is free and open-source software. See the original repository for license information.

## Citation

If you use this code in your research, please cite:

```
Ishihara, S., & Sugimura, K. (2012).
Bayesian inference of force dynamics during morphogenesis.
Journal of Theoretical Biology, 313, 201-211.
```

## Contact

For questions about the original Mathematica implementation, see Ali Hashmi's [Wolfram Community post](https://community.wolfram.com/groups/-/m/t/1571507).
