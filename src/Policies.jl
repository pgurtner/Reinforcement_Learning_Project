"""
    module Policies

Everything regarding policies, including the main struct, evaluation, updating and initialization functions.
"""
module Policies
using Random
using Distributions
using ..MDPs


"""
    mutable struct Policy

The central struct a policy consists of. 

This is mutable to allow efficient modifications in learning iterations.

# Fields
- `_probActions::Array{DiscreteDistribution, 2}`: array containing a probability distribution per state
"""
mutable struct Policy
    _probActions::Array{DiscreteDistribution, 2}
end

"""
    _assert_valid_state(π, state)

Helper function that checks that state is valid wrt. π.

# Parameters
- `π::Policy`:
- `state:Pair{Int}`:
"""
function _assert_valid_state(π::Policy, state::Pair{Int})
    dims = size(π._probActions)
    @assert 0 < state.first <= dims[1]
    @assert 0 < state.second <= dims[2]
end

"""
    evaluate(π, state)

Samples π(⋅|state), i.e. draws from the probability distribution of state.

# Parameters
- `π::Policy`:
- `state::Pair{Int}`:

# Returns
Sample from probability distribution of state in π.
"""
function evaluate(π::Policy, state::Pair{Int})::Int
    _assert_valid_state(π, state)

    return rand(π._probActions[state.first, state.second])
end

"""
    _extractProbability(d, _)

See comment in probActionGivenState.
"""
function _extractProbability(d::DiscreteUniform, _::Int)
    return d.pv
end

"""
    _extractProbability(d, action)

See comment in probActionGivenState.
"""
function _extractProbability(d::Categorical, action::Int)
    return d.p[action]
end

"""
    probActionGivenState(π, action, state)

# Returns
Gives π(action | state).
"""
function probActionGivenState(π::Policy, action::Int, state::Pair{Int})
    _assert_valid_state(π, state)

    # π._probActions contains (currently) distribution instances of DiscreteUniform and Categorical
    # these have different fields from which one can get individual probabilities, so this has to be
    # extracted with overloaded functions (or some dynamic type checking if Julia supports this)
    return _extractProbability(π._probActions[state.first, state.second], action)
end

"""
    updateState(π, state, Q, ϵ)

Updates π at state to be ϵ-greedy wrt. Q. Intended for use in the end of each iteration of learning algorithms.

This is a mutating function to avoid deleting and allocating arrays all the time because this gets called in every single learning iteration.

# Parameters
- `π::Policy`: respective policy
- `state::Pair{Int}`: state at which π gets updated
- `Q::Array{Float64, 3}`: action-value function according to which π is made ϵ-greedy
- `ϵ::Float64`: parameter for the ϵ-greedyness of π

# todo
- maybe remove asserts if they're too performance heavy
"""
function updateState!(π::Policy, state::Pair{Int}, Q::Array{Float64, 3}, ϵ::Float64)
    _assert_valid_state(π, state)

    # check that dimensions of π and Q match
    qDims = size(Q)
    policyDims = size(π._probActions)
    @assert qDims[1] == policyDims[1]
    @assert qDims[2] == policyDims[2]

    # setup ϵ-greedy distribution by setting all options to ϵ/numberOfOptions
    probabilities = fill(ϵ/qDims[3], qDims[3])
    
    # break ties uniformly by adding (1 - ϵ)/numberOfOptimalActions to all optimal actions
    # get max value
    maxValue = maximum(Q[state.first, state.second, :])
    # select optimal actions based on max value
    optimalActions = [action for action in MDPs.ACTIONS if Q[state.first, state.second, action] == maxValue]

    # update probabilities for optimal actions
    for a in optimalActions
        v = (1 - ϵ)/length(optimalActions)
        probabilities[a] += v
    end

    # create probability distribution based on probability vector and set it as the new value
    π._probActions[state.first, state.second] = Categorical(probabilities)
end

"""
    makeDeterministicPolicy(stateSpaceDims)

Instantiates and initializes policy to a deterministic policy that always picks the first action.

Currently not used.

# Parameters
- `stateSpaceDims::Pair{Int}`: width and height of the respective grid world

# Returns
Policy that always picks the first action.
"""
function makeDeterministicPolicy(stateSpaceDims::Pair{Int})::Policy
    @assert stateSpaceDims.first > 0
    @assert stateSpaceDims.second > 0

    # this uses the same distribution for every state, so create it once
    probabilities = zeros(Float64, length(MDPs.ACTIONS))
    probabilities[1] = 1.0
    distr = Categorical(probabilities)

    # and then create the action probabilities array by filling it with that distribution
    probActions = fill(distr, stateSpaceDims.first, stateSpaceDims.second)

    return Policy(probActions)
end

# todo: heavy code duplication to makeDeterministicPolicy
"""
    makeUniformlyRandomPolicy(stateSpaceDims)

instantiates and initializes policy to a uniform distribution for every state.

# Parameters
- `stateSpaceDims::Pair{Int}`: width and height of the respective grid world

# Returns
Policy that uniformly picks an action.
"""
function makeUniformlyRandomPolicy(stateSpaceDims::Pair{Int})::Policy
    @assert stateSpaceDims.first > 0
    @assert stateSpaceDims.second > 0

    # todo: this is very fragile when one should ever change MDPs.ACTIONS
    distr = DiscreteUniform(MDPs.ACTIONS[1], MDPs.ACTIONS[end])

    probActions = fill(distr, stateSpaceDims.first, stateSpaceDims.second)

    return Policy(probActions)
end

"""
    makeGreedyPolicy(Q)

Creates a greedy policy that deterministically picks the action with the highest value in the action-value function Q.

# Parameters
- `Q::Array{Float64, 3}`: some action-value function

# Returns
Policy that is greedy wrt. Q.
"""
function makeGreedyPolicy(Q::Array{Float64, 3})::Policy
    dims = size(Q)

    # initialize as undefined to allow index accesses
    probActions = Array{DiscreteDistribution, 2}(undef, dims[1], dims[2])

    # for every state
    for x in 1:dims[1]
        for y in 1:dims[2]
            # setup a zero vector
            p = zeros(dims[3])

            # pick optimal action
            a = argmax(Q[x, y, :])

            # set that action's probability to one
            p[a] = 1.0

            # and again create distribution based on a probability vector
            d =  Categorical(p)

            probActions[x, y] = d
        end
    end

    return Policy(probActions)
end

"""
    stateValueFunctionFromQ(π, Q)

Creates state-value function based on π and Q.

Used for benchmarking.
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
end