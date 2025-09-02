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

# Try to import Julia functions, fail if not available
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
    print(f" Julia not available: {e}")
    print("Please install Julia and PyJulia:")
    print("  1. Install Julia from https://julialang.org/downloads/")
    print("  2. pip install PyJulia")
    print("  3. python -c 'import julia; julia.install()'")
    sys.exit(1)
except Exception as e:
    print(f" Error loading Julia functions: {e}")
    print("Please ensure Julia is properly installed and configured")
    sys.exit(1)

class JuliaAcceleratedFingerprint:
    """
    High-performance wrapper for radial fingerprint computation using Julia acceleration.
    """
    
    def __init__(self, use_julia: bool = True):
        """
        Initialize the Julia-accelerated fingerprint computer.
        
        Args:
            use_julia: Whether to use Julia acceleration (always True if Julia available)
        """
        if not JULIA_AVAILABLE:
            print("❌ Julia is not available. Cannot initialize JuliaAcceleratedFingerprint.")
            sys.exit(1)
        
        self.use_julia = True  # Always use Julia if available
        print("Using Julia acceleration for fingerprint computation")
    
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
            # This should never happen since we exit if Julia is not available
            raise RuntimeError("Julia is not available")
    
    
    
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
            # This should never happen since we exit if Julia is not available
            raise RuntimeError("Julia is not available")
    
    
    
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
            # This should never happen since we exit if Julia is not available
            raise RuntimeError("Julia is not available")
    
    
    
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
            # This should never happen since we exit if Julia is not available
            raise RuntimeError("Julia is not available")
    
    
    
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
            # This should never happen since we exit if Julia is not available
            raise RuntimeError("Julia is not available")
    
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
            use_julia: Whether to use Julia acceleration (always True if Julia available)
        """
        self.params = None
        self.systems = []
        self.style = "radial"
        self.inputpath = inputpath
        self.dumppath = dumppath
        self.julia_accelerator = JuliaAcceleratedFingerprint(use_julia=use_julia)
    
    def input_parser(self):
        """Parse input parameters """
       
        pass
    
    def dump_parser(self):
        """Parse dump files """
        
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




    

