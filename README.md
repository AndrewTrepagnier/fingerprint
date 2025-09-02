
## Overview

This package provides a Python-to-Julia wrapper for computationally intensive parts of radial fingerprint computation. Structural fingerprints compress atomistic environment information into lower-dimensional inputs for neural networks, and are computed every timestep, making them performance-critical.


### Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Julia dependencies**:
   ```bash
   julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "PyCall", "Threads"])'
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Optional Performance Dependencies

For maximum performance, install additional Julia packages:
```bash
julia -e 'using Pkg; Pkg.add(["BenchmarkTools", "StaticArrays", "LoopVectorization"])'
```

## Usage

### Basic Usage

```python
from src.radial_julia_wrapper import JuliaAcceleratedFingerprint_radial

# Initialize with Julia acceleration
fingerprint_computer = JuliaAcceleratedFingerprint_radial(
    inputpath="path/to/input.txt",
    dumppath="path/to/dump/files",
    use_julia=True  # Set to False to use Python only
)

# Parse input parameters and dump files
fingerprint_computer.input_parser()
fingerprint_computer.dump_parser()

# Compute fingerprints
result = fingerprint_computer.compute_fingerprint(system_index=0)
summed_fingerprints, individual_fingerprints = result[:-1], result[-1]
```

### Direct Function Usage

```python
from src.radial_julia_wrapper import JuliaAcceleratedFingerprint
import numpy as np

# Initialize accelerator
accelerator = JuliaAcceleratedFingerprint(use_julia=True)

# Create test data
positions = np.random.rand(100, 3) * 10.0
num_atoms = 100

# Compute distance matrix
distance_matrix = accelerator.compute_distance_matrix(positions, num_atoms)

# Compute cutoff function
cutoff_value = accelerator.cutoff_function(r=3.0, rc=6.0, dr=0.5)

# Generate radii table
r1, radii_table, dfctable = accelerator.radii_table(
    rc=6.0, dr=0.5, re=2.0, n=8.0, o=0.0, 
    alphak=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
)

# Compute fingerprints
summed_fps, individual_fps = accelerator.compute_fingerprint(
    distance_matrix, num_atoms, r1, radii_table, 8
)
```

### Batch Processing

```python
# Process multiple systems efficiently
positions_list = [np.random.rand(100, 3) * 10.0 for _ in range(10)]
params = {
    "rc": 6.0, "dr": 0.5, "re": 2.0, "n": 8.0, "o": 0.0,
    "alphak": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
}

results = accelerator.compute_fingerprints_batch(positions_list, params)
```

## Performance Comparison

Run the built-in benchmark to compare Python vs Julia performance:

```python
from src.radial_julia_wrapper import benchmark_performance
benchmark_performance()
```

Typical results:
- **Distance Matrix**: 5-20x speedup
- **Radii Table**: 10-50x speedup  
- **Fingerprint Computation**: 20-100x speedup
- **Overall**: 10-50x speedup depending on system size

## Architecture

### Julia Functions (`src/radial_fingerprint.jl`)

- `compute_distance_matrix_julia`: O(n²) distance matrix computation
- `cutoff_function_julia`: Smooth cutoff function evaluation
- `radii_table_julia`: Pre-computed interpolation tables
- `compute_fingerprint_julia`: Catmull-Rom spline interpolation
- `compute_fingerprints_batch_julia`: Batch processing

### Python Wrapper (`src/radial_julia_wrapper.py`)

- `JuliaAcceleratedFingerprint`: Core wrapper class
- `JuliaAcceleratedFingerprint_radial`: Drop-in replacement for original class
- Automatic fallback to Python implementation
- Type conversion and error handling

## Integration with Existing Code

The wrapper is designed as a drop-in replacement for the original `Fingerprint_radial` class:

```python
# Original code
from src.radial import Fingerprint_radial
fingerprint = Fingerprint_radial(inputpath, dumppath)

# Julia-accelerated version
from src.radial_julia_wrapper import JuliaAcceleratedFingerprint_radial
fingerprint = JuliaAcceleratedFingerprint_radial(inputpath, dumppath, use_julia=True)
```

## Troubleshooting

### Julia Not Available

If Julia is not installed or PyJulia fails to load:
- The wrapper automatically falls back to Python implementation
- Check Julia installation: `julia --version`
- Install PyJulia: `pip install PyJulia`
- Configure PyJulia: `python -c "import julia; julia.install()"`

### Performance Issues

- Ensure Julia is compiled: First run may be slower
- Use `@inbounds` and `@simd` in Julia for additional optimization
- Consider using `StaticArrays` for small systems
- Enable threading: `export JULIA_NUM_THREADS=4`

### Memory Issues

- Large systems may require more memory
- Consider processing in batches
- Monitor memory usage with `@allocated` in Julia

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure both Python and Julia implementations give identical results
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{radial_fingerprint_julia,
  title={Julia-Accelerated Radial Fingerprint for Machine Learned Interatomic Potentials},
  author={Andrew Trepagnier},
  year={2024},
  url={https://github.com/your-repo/radial-fingerprint-julia}
}
```

## Contact

Andrew Trepagnier (MSU) | andrew.trepagnier1@gmail.com
