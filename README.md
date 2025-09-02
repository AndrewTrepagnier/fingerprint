
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


