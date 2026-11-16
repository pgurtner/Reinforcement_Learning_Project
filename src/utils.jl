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
end