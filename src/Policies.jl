module Policies

export Policy

module GridWorld
    using Random
    using Distributions
    using ...MDPs

    export Policy, evaluate, updateState!, makeDeterministicPolicy, makeUniformlyRandomPolicy

    """
        mutable struct Policy

    todo

    This is mutable to allow efficient modifications in learning iterations.

    """
    mutable struct Policy
        _probActions::Array{DiscreteDistribution, 2}
    end

    function evaluate(π::Policy, state::Pair{Int})::Int
        dims = size(π._probActions)
        @assert 0 < state.first <= dims[1]
        @assert 0 < state.second <= dims[2]

        return rand(π._probActions[state.first, state.second])
    end

    """
        setToQEpsGreedy!(π, Q, ϵ)

    todo

    This is a mutating function to avoid deleting and allocating arrays all the time because this gets called in every single learning iteration.

    # todo
    - maybe remove asserts if they're too performance heavy
    """
    function updateState!(π::Policy, state::Pair{Int}, Q::Array{Float64, 3}, ϵ::Float64)
        policyDims = size(π._probActions)
        @assert 0 < state.first <= policyDims[1]
        @assert 0 < state.second <= policyDims[2]

        qDims = size(Q)
        @assert qDims[1] == policyDims[1]
        @assert qDims[2] == policyDims[2]

        probabilities = fill(ϵ, qDims[3])
        a = argmax(Q[state.first, state.second, :])
        probabilities[a] = 1.0 - ((qDims[3] - 1) * ϵ)

        π._probActions[state.first, state.second] = Categorical(probabilities)
    end

    function makeDeterministicPolicy(stateSpaceDims::Pair{Int})::Policy
        @assert stateSpaceDims.first > 0
        @assert stateSpaceDims.second > 0
        @assert length(actions) > 0

        probabilities = zeros(Float64, length(MDPs.GridWorld.ACTIONS))
        probabilities[1] = 1.0
        distr = Categorical(probabilities)

        probActions = fill(distr, stateSpaceDims.first, stateSpaceDims.second)

        return Policy(probActions)
    end

    # todo: heavy code duplication to makeDeterministicPolicy
    function makeUniformlyRandomPolicy(stateSpaceDims::Pair{Int})::Policy
        @assert stateSpaceDims.first > 0
        @assert stateSpaceDims.second > 0

        distr = DiscreteUniform(MDPs.GridWorld.ACTIONS[1], MDPs.GridWorld.ACTIONS[end])

        probActions = fill(distr, stateSpaceDims.first, stateSpaceDims.second)

        return Policy(probActions)
    end
end

end