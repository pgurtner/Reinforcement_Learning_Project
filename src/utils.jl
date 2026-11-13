module Utils
using LinearAlgebra: I

function stdbasis(l::Int, i::Int)
    return Float64.(I[1:l, i])
end


end