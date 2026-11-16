module Policies

using Random
using Distributions
using ..MDPs

export Policy, evaluate, updateState!, makeDeterministicPolicy, makeUniformlyRandomPolicy

"""
    mutable struct Policy

todo

This is mutable to allow efficient modifications in learning iterations.

"""
mutable struct Policy
    _probActions::Array{DiscreteDistribution, 2}
end

function _assert_valid_state(π::Policy, state::Pair{Int})
    dims = size(π._probActions)
    @assert 0 < state.first <= dims[1]
    @assert 0 < state.second <= dims[2]
end

"""
todo
"""
function evaluate(π::Policy, state::Pair{Int})::Int
    _assert_valid_state(π, state)

    return rand(π._probActions[state.first, state.second])
end

function _extractProbability(d::DiscreteUniform, _::Int)
    return d.pv
end

function _extractProbability(d::Categorical, action::Int)
    return d.p[action]
end

"""
"""
function probActionGivenState(π::Policy, action::Int, state::Pair{Int})
    _assert_valid_state(π, state)

    # π._probActions contains (currently) distribution instances of DiscreteUniform and Categorical
    # these have different fields from which one can get individual probabilities, so this has to be
    # extracted with overloaded functions (or some dynamic type checking if Julia supports this)
    return _extractProbability(π._probActions[state.first, state.second], action)
end

"""
    setToQEpsGreedy!(π, Q, ϵ)

todo

This is a mutating function to avoid deleting and allocating arrays all the time because this gets called in every single learning iteration.

# todo
- maybe remove asserts if they're too performance heavy
"""
function updateState!(π::Policy, state::Pair{Int}, Q::Array{Float64, 3}, ϵ::Float64)
    _assert_valid_state(π, state)

    qDims = size(Q)
    policyDims = size(π._probActions)
    @assert qDims[1] == policyDims[1]
    @assert qDims[2] == policyDims[2]


    probabilities = fill(ϵ/qDims[3], qDims[3])
    
    # break ties uniformly
    maxValue = maximum(Q[state.first, state.second, :])
    optimalActions = [action for action in MDPs.ACTIONS if Q[state.first, state.second, action] == maxValue]
    for a in optimalActions
        v = (1 - ϵ)/length(optimalActions)
        probabilities[a] += v
    end

    π._probActions[state.first, state.second] = Categorical(probabilities)
end

function makeDeterministicPolicy(stateSpaceDims::Pair{Int})::Policy
    @assert stateSpaceDims.first > 0
    @assert stateSpaceDims.second > 0

    probabilities = zeros(Float64, length(MDPs.ACTIONS))
    probabilities[1] = 1.0
    distr = Categorical(probabilities)

    probActions = fill(distr, stateSpaceDims.first, stateSpaceDims.second)

    return Policy(probActions)
end

# todo: heavy code duplication to makeDeterministicPolicy
function makeUniformlyRandomPolicy(stateSpaceDims::Pair{Int})::Policy
    @assert stateSpaceDims.first > 0
    @assert stateSpaceDims.second > 0

    distr = DiscreteUniform(MDPs.ACTIONS[1], MDPs.ACTIONS[end])

    probActions = fill(distr, stateSpaceDims.first, stateSpaceDims.second)

    return Policy(probActions)
end

function makeGreedyPolicy(Q::Array{Float64, 3})::Policy
    dims = size(Q)

    probActions = Array{DiscreteDistribution, 2}(undef, dims[1], dims[2])

    for x in 1:dims[1]
        for y in 1:dims[2]
            p = zeros(dims[3])
            a = argmax(Q[x, y, :])
            p[a] = 1.0
            d =  Categorical(p)

            probActions[x, y] = d
        end
    end

    return Policy(probActions)
end
end