"""
Julia implementation of radial bond structural fingerprint for high-performance computation.
This module provides the computationally intensive parts that can be called from Python.

Author: Andrew Trepagnier (MSU) | andrew.trepagnier1@gmail.com
"""

using LinearAlgebra
using PyCall

# Export functions that will be called from Python
export compute_distance_matrix_julia, cutoff_function_julia, radii_table_julia, compute_fingerprint_julia

"""
Compute distance matrix between all atoms efficiently in Julia.

Args:
    positions: Array of shape (n_atoms, 3) containing atomic positions
    num_atoms: Number of atoms in the system

Returns:
    Distance matrix of shape (n_atoms, n_atoms)
"""
function compute_distance_matrix_julia(positions::Array{Float64,2}, num_atoms::Int)
    distance_matrix = zeros(num_atoms, num_atoms)
    
    for i in 1:num_atoms
        for j in 1:num_atoms
            if i != j
                distance_matrix[i,j] = norm(positions[i,:] - positions[j,:])
            end
        end
    end
    
    return distance_matrix
end

"""
Compute cutoff function for a given radius.

Args:
    r: Distance for which to compute cutoff function
    rc: Cutoff radius
    dr: Radial step size

Returns:
    Cutoff function value between 0 and 1
"""
function cutoff_function_julia(r::Float64, rc::Float64, dr::Float64)
    x = (rc - r) / dr
    if x > 1.0
        return 1.0
    elseif 0.0 <= x <= 1.0
        return (1.0 - (1.0 - x)^4)^2
    else
        return 0.0
    end
end

"""
Generate interpolation tables of radii and derivatives for computational efficiency.

Args:
    rc: Cutoff radius
    dr: Radial step size
    re: Equilibrium distance
    n: Power series upper bound
    o: Power series lower bound
    alphak: Array of alphak parameters

Returns:
    Tuple of (r1, radii_table, dfctable)
"""
function radii_table_julia(rc::Float64, dr::Float64, re::Float64, n::Float64, o::Float64, alphak::Array{Float64,1})
    buffer = 5
    res = 1000
    
    num_fingerprints = Int(n - o)
    radii_table = zeros(res + buffer, num_fingerprints)
    dfctable = zeros(res + buffer)
    r1 = zeros(res + buffer)
    
    for m in 1:num_fingerprints
        for k in 1:(res + buffer)
            r1[k] = rc * (k - 1) / res
            r1_sqrt = sqrt(r1[k])
            
            term_of_mth_fp = (
                (r1_sqrt / re)^(m-1) *
                exp(-alphak[m] * (r1_sqrt / re)) *
                cutoff_function_julia(r1_sqrt, rc, dr)
            )
            radii_table[k, m] = term_of_mth_fp
            
            if r1_sqrt >= rc || r1_sqrt <= rc - dr
                dfctable[k] = 0.0
            else
                term = (rc - r1_sqrt) / dr
                dfctable[k] = (-8.0 * (1.0 - term)^3) / (dr * (1.0 - term)^4)
            end
        end
    end
    
    return r1, radii_table, dfctable
end

"""
Catmull-Rom spline interpolation for smooth fingerprint computation.

Args:
    t: Interpolation parameter (0 to 1)
    y: Array of 4 control points

Returns:
    Interpolated value
"""
function catmull_rom_interpolate(t::Float64, y::Array{Float64,1})
    t2 = t * t
    t3 = t2 * t
    
    # Catmull-Rom coefficients
    p0 = -0.5*t3 + t2 - 0.5*t
    p1 = 1.5*t3 - 2.5*t2 + 1.0
    p2 = -1.5*t3 + 2.0*t2 + 0.5*t
    p3 = 0.5*t3 - 0.5*t2
    
    return y[1] * p0 + y[2] * p1 + y[3] * p2 + y[4] * p3
end

"""
Compute fingerprints for a given atomic system using Catmull-Rom spline interpolation.

Args:
    distance_matrix: Pre-computed distance matrix
    num_atoms: Number of atoms
    r1: Pre-computed radius table
    radii_table: Pre-computed radii table
    num_fingerprints: Number of fingerprint types

Returns:
    Tuple of (summed_fingerprints, individual_fingerprints)
"""
function compute_fingerprint_julia(
    distance_matrix::Array{Float64,2}, 
    num_atoms::Int, 
    r1::Array{Float64,1}, 
    radii_table::Array{Float64,2}, 
    num_fingerprints::Int
)
    fingerprints = zeros(num_atoms, num_fingerprints)
    
    for i in 1:num_atoms
        for j in 1:num_atoms
            if i != j
                rij = distance_matrix[i,j]
                idx = searchsortedfirst(r1, rij) - 1
                
                if idx > 1 && idx < length(r1) - 2
                    t = (rij - r1[idx]) / (r1[idx+1] - r1[idx])
                    
                    for m in 1:num_fingerprints
                        y = [
                            radii_table[idx-1, m],
                            radii_table[idx, m],
                            radii_table[idx+1, m],
                            radii_table[idx+2, m]
                        ]
                        
                        interpolated_value = catmull_rom_interpolate(t, y)
                        fingerprints[i,m] += interpolated_value
                    end
                end
            end
        end
    end
    
    # Calculate sums for each fingerprint type
    summed_fingerprints = [sum(fingerprints[:, m]) for m in 1:num_fingerprints]
    
    return summed_fingerprints, fingerprints
end

"""
High-performance batch computation of fingerprints for multiple systems.

Args:
    positions_list: List of position arrays
    params: Dictionary of parameters (rc, dr, re, n, o, alphak)

Returns:
    List of (summed_fingerprints, individual_fingerprints) tuples
"""
function compute_fingerprints_batch_julia(positions_list::Array{Array{Float64,2},1}, params::Dict)
    results = []
    
    # Pre-compute tables once for all systems
    r1, radii_table, dfctable = radii_table_julia(
        params["rc"], params["dr"], params["re"], 
        params["n"], params["o"], params["alphak"]
    )
    
    num_fingerprints = Int(params["n"] - params["o"])
    
    for positions in positions_list
        num_atoms = size(positions, 1)
        distance_matrix = compute_distance_matrix_julia(positions, num_atoms)
        
        summed_fps, individual_fps = compute_fingerprint_julia(
            distance_matrix, num_atoms, r1, radii_table, num_fingerprints
        )
        
        push!(results, (summed_fps, individual_fps))
    end
    
    return results
end

# Optional: Add type annotations for better performance
const Float64Array = Array{Float64,2}
const Float64Vector = Array{Float64,1}

# Performance optimizations
@inline function cutoff_function_julia_inline(r::Float64, rc::Float64, dr::Float64)
    x = (rc - r) / dr
    if x > 1.0
        return 1.0
    elseif 0.0 <= x <= 1.0
        return (1.0 - (1.0 - x)^4)^2
    else
        return 0.0
    end
end

# Threaded version for multi-core systems
using Threads

"""
Threaded version of distance matrix computation for better performance on multi-core systems.
"""
function compute_distance_matrix_julia_threaded(positions::Array{Float64,2}, num_atoms::Int)
    distance_matrix = zeros(num_atoms, num_atoms)
    
    Threads.@threads for i in 1:num_atoms
        for j in 1:num_atoms
            if i != j
                distance_matrix[i,j] = norm(positions[i,:] - positions[j,:])
            end
        end
    end
    
    return distance_matrix
end
