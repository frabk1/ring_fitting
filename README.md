# ringfit

Tools for extracting, geometrically fitting and comparing rings data, focused on black hole N=1 photon ring images.

This package is under active development. If you encounter issues or have feature requests, please open an issue or submit a pull request.

The functionality centers on extracting bright ring structures from images and fitting geometric models to characterize ring radius, thickness, and asymmetry.

## Installation

### Install from PyPI

The latest release is available on PyPI:

    pip install ringfit

### Development installation

For development or working with the latest code:

    git clone https://github.com/frabk1/ring_fitting.git
    cd ring_fitting
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .

## Quickstart
```python
    import ehtim as eh
    from ringfit.extraction import estimate_ring_parameters, rbp_find_bright_points
    from ringfit.fitting import general_fit

    # Load an eht-imaging FITS file
    im = eh.image.load_fits("path/to/image.fits")

    # Estimate ring parameters and center
    radius, width, peak, background, (xc, yc) = estimate_ring_parameters(im)

    # Extract bright points using recursive brightest point search
    points = rbp_find_bright_points(im, radius=radius)

    # Fit a geometric ring model, circle shown here
    result = general_fit(points, xc, yc, shape="circle")

    print(result)
```
## Repository structure

    ringfit/        Core library code
    examples/       Example scripts and workflows
    tests/          Unit tests
    notebooks/      Exploratory notebooks

## Dependencies

ringfit is designed to load images with eht-imaging and assumes images are fits files.

Most functionality relies only on standard scientific Python packages (numpy, scipy, matplotlib). Additional dependencies may be required for specific projects.

## Citation

If you use this software in academic work, please cite the associated JOSS paper when available.

A machine-readable citation is also provided in CITATION.cff.

## License

MIT License. See the LICENSE file.
