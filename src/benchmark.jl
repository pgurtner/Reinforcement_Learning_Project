"""
    module Benchmark

The module containing everything around running benchmarks. Plotting is handled by the module Plots.

# ideas
- convergence
- performance compared to other approaches
- performance by itself??
- accuracy

3. implement benchmarking (something like figure 7.2, between algorithms for efficiency and by oneself for convergence, IMPORTANT: make sure not to mix up state-value and action-value functions, look again into how you can get the state-value function from the action-value function)

"""
module Benchmark
using ..Policies
using ..MDPs

"""
todo
"""
function stateValueFunctionFromQ(π::Policies.Policy, Q::Array{Float64, 3})::Array{Float64, 2}
    # todo: assert matching dimensions

    dims = size(Q)
    v = zeros(dims[1], dims[2])

    # todo this can probably be written with element-wise operations
    for x in 1:dims[1]
        for y in 1:dims[2]
            s = Pair(x, y)
            for a in MDPs.ACTIONS
                v[x, y] += Policies.probActionGivenState(π, a, s) * Q[x, y, a]
            end
        end
    end

    return v
end

"""
todo
"""
function exactActionValueFunction(mdp::MDPs.MDPConfig, γ::Float64)
    dims = MDPs.dimensions(mdp)
    Q = zeros(dims[1], dims[2], length(MDPs.ACTIONS))

    for a in MDPs.ACTIONS
        # todo: access to hidden field
        destinations = findall(v -> v > 0, mdp._rewards[:, :, a])

        for d in destinations
            dx, dy = d.I
            for x in 1:dims[1]
                for y in 1:dims[2]
                    if MDPs.is_terminal(mdp, Pair(x, y))
                        continue
                    end

                    # todo: access to hidden field
                    Q[x, y, a] += mdp._rewards[d, a] * γ^(abs(dx - x) + abs(dy - y))
                end
            end
        end
    end

    return Q
end

"""
todo
Compute exact state-value function for mdp for error computations
"""
function exactStateValueFunction(Q::Array{Float64, 3})::Array{Float64, 2}
    π = Policies.makeGreedyPolicy(Q)
    V = stateValueFunctionFromQ(π, Q)
    return V
end

"""
todo
"""
function meanSquaredError(A::Array{Float64, N}, B::Array{Float64, N}) where N <: Int
    @assert size(A) == size(B)
    dims = size(A)
    n = prod(dims)
     
    return 1/n * sum((A .- B) .^ 2)
end

"""
todo
"""
function rms(A::Array{Float64, N}, B::Array{Float64, N}) where N <: Int
    return sqrt.(meanSquaredError(A, B))
end

end