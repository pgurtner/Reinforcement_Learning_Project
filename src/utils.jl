"""
    module Utils

General utility module that contains helper functions that don't belong to a specific module.
"""
module Utils
using LinearAlgebra: I

"""
    stdbasis(l, i)

Gives e_i of R^l.
"""
function stdbasis(l::Int, i::Int)
    # I is an abstract object that represents the identity matrix of any size.
    # By selecting the range 1:l in the first index, its overall size is set.
    # Accessing the ith element then means to pick the ith column of the identity matrix of size 1:l in each dimension.
    return Float64.(I[1:l, i])
end

"""
    meanSquaredError(A, B)

Computes 1/n * sum_i((A[i] - B[i])^2) for arrays A and B of same size.
"""
function meanSquaredError(A::Array{Float64, N}, B::Array{Float64, N}) where {N}
    @assert size(A) == size(B)
    dims = size(A)
    n = prod(dims)
     
    return 1/n * sum((A .- B) .^ 2)
end

"""
    rms(A, B)

Computes root mean squared error, i.e. sqrt.(meanSquaredError(A, B)) for arrays A and B of same size.

Currently unused.
"""
function rms(A::Array{Float64, N}, B::Array{Float64, N}) where {N}
    return sqrt.(meanSquaredError(A, B))
end
end