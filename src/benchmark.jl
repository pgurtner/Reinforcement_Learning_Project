"""
    module Benchmark

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

end