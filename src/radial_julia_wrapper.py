"""
Python wrapper for Julia-accelerated radial fingerprint computation.
This module provides high-performance alternatives to the computationally intensive
parts of the radial fingerprint code by calling optimized Julia functions.

Author: Andrew Trepagnier (MSU) | andrew.trepagnier1@gmail.com
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union
import os
import sys

# Try to import Julia functions, fall back to Python if not available
try:
    import julia
    from julia import Main
    from julia import LinearAlgebra
    from julia import Threads
    
    # Add the src directory to Julia's load path
    julia_src_path = os.path.join(os.path.dirname(__file__), 'radial_fingerprint.jl')
    Main.include(julia_src_path)
    
    JULIA_AVAILABLE = True
    print("Julia acceleration available")
except ImportError as e:
    print(f"Julia not available, falling back to Python: {e}")
    JULIA_AVAILABLE = False
except Exception as e:
    print(f"Error loading Julia functions: {e}")
    JULIA_AVAILABLE = False

class JuliaAcceleratedFingerprint:
    """
    High-performance wrapper for radial fingerprint computation using Julia acceleration.
    """
    
    def __init__(self, use_julia: bool = True):
        """
        Initialize the Julia-accelerated fingerprint computer.
        
        Args:
            use_julia: Whether to use Julia acceleration (falls back to Python if Julia unavailable)
        """
        self.use_julia = use_julia and JULIA_AVAILABLE
        if self.use_julia:
            print("Using Julia acceleration for fingerprint computation")
        else:
            print("Using Python implementation for fingerprint computation")
    
    def compute_distance_matrix(self, positions: np.ndarray, num_atoms: int) -> np.ndarray:
        """
        Compute distance matrix between all atoms.
        
        Args:
            positions: Array of shape (n_atoms, 3) containing atomic positions
            num_atoms: Number of atoms in the system
            
        Returns:
            Distance matrix of shape (n_atoms, n_atoms)
        """
        if self.use_julia:
            # Convert to Julia-compatible format
            positions_julia = np.asarray(positions, dtype=np.float64)
            return Main.compute_distance_matrix_julia(positions_julia, num_atoms)
        else:
            # Fallback to Python implementation
            return self._compute_distance_matrix_python(positions, num_atoms)
    
    def _compute_distance_matrix_python(self, positions: np.ndarray, num_atoms: int) -> np.ndarray:
        """Python fallback for distance matrix computation."""
        distance_matrix = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    distance_matrix[i,j] = np.linalg.norm(positions[i] - positions[j])
        return distance_matrix
    
    def cutoff_function(self, r: float, rc: float, dr: float) -> float:
        """
        Compute cutoff function for a given radius.
        
        Args:
            r: Distance for which to compute cutoff function
            rc: Cutoff radius
            dr: Radial step size
            
        Returns:
            Cutoff function value between 0 and 1
        """
        if self.use_julia:
            return Main.cutoff_function_julia(float(r), float(rc), float(dr))
        else:
            return self._cutoff_function_python(r, rc, dr)
    
    def _cutoff_function_python(self, r: float, rc: float, dr: float) -> float:
        """Python fallback for cutoff function."""
        x = (rc - r) / dr
        if x > 1:
            return 1
        elif 0 <= x <= 1:
            return (1 - (1 - x)**4)**2
        else:
            return 0
    
    def radii_table(self, rc: float, dr: float, re: float, n: float, o: float, alphak: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate interpolation tables of radii and derivatives.
        
        Args:
            rc: Cutoff radius
            dr: Radial step size
            re: Equilibrium distance
            n: Power series upper bound
            o: Power series lower bound
            alphak: List of alphak parameters
            
        Returns:
            Tuple of (r1, radii_table, dfctable)
        """
        if self.use_julia:
            alphak_julia = np.array(alphak, dtype=np.float64)
            r1, radii_table, dfctable = Main.radii_table_julia(
                float(rc), float(dr), float(re), float(n), float(o), alphak_julia
            )
            return r1, radii_table, dfctable
        else:
            return self._radii_table_python(rc, dr, re, n, o, alphak)
    
    def _radii_table_python(self, rc: float, dr: float, re: float, n: float, o: float, alphak: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Python fallback for radii table computation."""
        buffer = 5
        res = 1000
        
        num_fingerprints = int(n - o)
        radii_table = np.zeros((res + buffer, num_fingerprints))
        dfctable = np.zeros(res + buffer)
        r1 = np.zeros(res + buffer)
        
        for m in range(num_fingerprints):
            for k in range(res + buffer):
                r1[k] = rc * k / res
                r1_sqrt = np.sqrt(r1[k])
                
                term_of_mth_fp = (
                    (r1_sqrt / re) ** m *
                    np.exp(-alphak[m] * (r1_sqrt / re)) *
                    self._cutoff_function_python(r1_sqrt, rc, dr)
                )
                radii_table[k, m] = term_of_mth_fp
                
                if r1_sqrt >= rc or r1_sqrt <= rc - dr:
                    dfctable[k] = 0
                else:
                    term = (rc - r1_sqrt) / dr
                    dfctable[k] = (-8 * (1 - term) ** 3) / (dr * (1 - term) ** 4)
        
        return r1, radii_table, dfctable
    
    def compute_fingerprint(self, distance_matrix: np.ndarray, num_atoms: int, r1: np.ndarray, 
                          radii_table: np.ndarray, num_fingerprints: int) -> Tuple[List[float], np.ndarray]:
        """
        Compute fingerprints using Catmull-Rom spline interpolation.
        
        Args:
            distance_matrix: Pre-computed distance matrix
            num_atoms: Number of atoms
            r1: Pre-computed radius table
            radii_table: Pre-computed radii table
            num_fingerprints: Number of fingerprint types
            
        Returns:
            Tuple of (summed_fingerprints, individual_fingerprints)
        """
        if self.use_julia:
            distance_matrix_julia = np.asarray(distance_matrix, dtype=np.float64)
            r1_julia = np.asarray(r1, dtype=np.float64)
            radii_table_julia = np.asarray(radii_table, dtype=np.float64)
            
            summed_fps, individual_fps = Main.compute_fingerprint_julia(
                distance_matrix_julia, num_atoms, r1_julia, radii_table_julia, num_fingerprints
            )
            return list(summed_fps), individual_fps
        else:
            return self._compute_fingerprint_python(distance_matrix, num_atoms, r1, radii_table, num_fingerprints)
    
    def _compute_fingerprint_python(self, distance_matrix: np.ndarray, num_atoms: int, r1: np.ndarray, 
                                  radii_table: np.ndarray, num_fingerprints: int) -> Tuple[List[float], np.ndarray]:
        """Python fallback for fingerprint computation."""
        fingerprints = np.zeros((num_atoms, num_fingerprints))
        
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    rij = distance_matrix[i,j]
                    idx = np.searchsorted(r1, rij) - 1
                    
                    if idx > 0 and idx < len(r1) - 2:
                        t = (rij - r1[idx]) / (r1[idx+1] - r1[idx])
                        
                        for m in range(num_fingerprints):
                            y = [
                                radii_table[idx-1, m],
                                radii_table[idx, m],
                                radii_table[idx+1, m],
                                radii_table[idx+2, m]
                            ]
                            
                            # Catmull-Rom interpolation
                            t2 = t * t
                            t3 = t2 * t
                            
                            p0 = -0.5*t3 + t2 - 0.5*t
                            p1 = 1.5*t3 - 2.5*t2 + 1.0
                            p2 = -1.5*t3 + 2.0*t2 + 0.5*t
                            p3 = 0.5*t3 - 0.5*t2
                            
                            interpolated_value = (
                                y[0] * p0 + 
                                y[1] * p1 + 
                                y[2] * p2 + 
                                y[3] * p3
                            )
                            
                            fingerprints[i,m] += interpolated_value
        
        summed_fingerprints = [np.sum(fingerprints[:, m]) for m in range(num_fingerprints)]
        return summed_fingerprints, fingerprints
    
    def compute_fingerprints_batch(self, positions_list: List[np.ndarray], params: Dict) -> List[Tuple[List[float], np.ndarray]]:
        """
        High-performance batch computation of fingerprints for multiple systems.
        
        Args:
            positions_list: List of position arrays
            params: Dictionary of parameters (rc, dr, re, n, o, alphak)
            
        Returns:
            List of (summed_fingerprints, individual_fingerprints) tuples
        """
        if self.use_julia:
            # Convert positions to Julia-compatible format
            positions_julia = [np.asarray(pos, dtype=np.float64) for pos in positions_list]
            return Main.compute_fingerprints_batch_julia(positions_julia, params)
        else:
            return self._compute_fingerprints_batch_python(positions_list, params)
    
    def _compute_fingerprints_batch_python(self, positions_list: List[np.ndarray], params: Dict) -> List[Tuple[List[float], np.ndarray]]:
        """Python fallback for batch fingerprint computation."""
        results = []
        
        # Pre-compute tables once for all systems
        r1, radii_table, dfctable = self._radii_table_python(
            params["rc"], params["dr"], params["re"], 
            params["n"], params["o"], params["alphak"]
        )
        
        num_fingerprints = int(params["n"] - params["o"])
        
        for positions in positions_list:
            num_atoms = positions.shape[0]
            distance_matrix = self._compute_distance_matrix_python(positions, num_atoms)
            
            summed_fps, individual_fps = self._compute_fingerprint_python(
                distance_matrix, num_atoms, r1, radii_table, num_fingerprints
            )
            
            results.append((summed_fps, individual_fps))
        
        return results

# Integration with existing Fingerprint_radial class
class JuliaAcceleratedFingerprint_radial:
    """
    Julia-accelerated version of the original Fingerprint_radial class.
    This class provides the same interface as the original but uses Julia acceleration
    for computationally intensive operations.
    """
    
    def __init__(self, inputpath: Optional[str] = None, dumppath: Optional[str] = None, use_julia: bool = True):
        """
        Initialize the Julia-accelerated fingerprint computer.
        
        Args:
            inputpath: Path to input parameter file
            dumppath: Path to dump files directory
            use_julia: Whether to use Julia acceleration
        """
        self.params = None
        self.systems = []
        self.style = "radial"
        self.inputpath = inputpath
        self.dumppath = dumppath
        self.julia_accelerator = JuliaAcceleratedFingerprint(use_julia=use_julia)
    
    def input_parser(self):
        """Parse input parameters (same as original)."""
        # This method remains the same as in the original class
        # You can copy the implementation from the original radial.py
        pass
    
    def dump_parser(self):
        """Parse dump files (same as original)."""
        # This method remains the same as in the original class
        # You can copy the implementation from the original radial.py
        pass
    
    def compute_fingerprint(self, system_index: int = 0) -> Tuple[float, ...]:
        """
        Compute fingerprints using Julia acceleration.
        
        Args:
            system_index: Index of the system to compute fingerprints for
            
        Returns:
            Tuple: (summed_F0, summed_F1, ..., summed_Fm, fingerprints)
        """
        if not self.systems:
            raise ValueError("No atomic systems loaded. Run dump_parser first.")
        
        system = self.systems[system_index]
        
        # Use Julia acceleration for distance matrix computation
        if system.distance_matrix is None:
            system.distance_matrix = self.julia_accelerator.compute_distance_matrix(
                system.atom_positions, system.num_atoms
            )
        
        # Use Julia acceleration for radii table computation
        r1, radii_table, dfctable = self.julia_accelerator.radii_table(
            self.params.rc, self.params.dr, self.params.re,
            self.params.n, self.params.o, self.params.alphak
        )
        
        # Use Julia acceleration for fingerprint computation
        num_fingerprints = int(self.params.n - self.params.o)
        summed_fingerprints, fingerprints = self.julia_accelerator.compute_fingerprint(
            system.distance_matrix, system.num_atoms, r1, radii_table, num_fingerprints
        )
        
        system.fingerprints = fingerprints
        
        return (*summed_fingerprints, fingerprints)

# Example usage and performance comparison
def benchmark_performance():
    """
    Benchmark the performance difference between Python and Julia implementations.
    """
    print("Performance Benchmark")
    print("=" * 50)
    
    # Create test data
    num_atoms = 100
    positions = np.random.rand(num_atoms, 3) * 10.0
    
    # Test parameters
    params = {
        "rc": 6.0,
        "dr": 0.5,
        "re": 2.0,
        "n": 8.0,
        "o": 0.0,
        "alphak": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    }
    
    # Test Python implementation
    python_accelerator = JuliaAcceleratedFingerprint(use_julia=False)
    
    import time
    start_time = time.time()
    distance_matrix_py = python_accelerator.compute_distance_matrix(positions, num_atoms)
    python_time = time.time() - start_time
    
    print(f"Python distance matrix computation: {python_time:.4f} seconds")
    
    # Test Julia implementation (if available)
    if JULIA_AVAILABLE:
        julia_accelerator = JuliaAcceleratedFingerprint(use_julia=True)
        
        start_time = time.time()
        distance_matrix_jl = julia_accelerator.compute_distance_matrix(positions, num_atoms)
        julia_time = time.time() - start_time
        
        print(f"Julia distance matrix computation: {julia_time:.4f} seconds")
        print(f"Speedup: {python_time / julia_time:.2f}x")
        
        # Verify results are the same
        if np.allclose(distance_matrix_py, distance_matrix_jl):
            print("✓ Results match between Python and Julia implementations")
        else:
            print("✗ Results differ between Python and Julia implementations")

if __name__ == "__main__":
    benchmark_performance()
