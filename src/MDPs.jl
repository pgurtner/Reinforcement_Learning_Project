"""
    module MDPs

todo: description

# todo
- the module name is a bit shit
"""
module MDPs

using Random
using Distributions
using ...Utils: stdbasis

export makeSimpleGridworld, transition, is_terminal, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, dimensions

"""
    MDPConfig(probActions, rewards)

todo

This struct is meant to be instantiated from this module and only handed to functions of this module without accessing its fields from the caller side.

# Fields
- `_probStates::Array{Float64, 5}`: todo
- `_rewards::Array{Float64, 3}`: todo
- `_nonBoundaryTerminalStates::Vector{Pair{Int}}`: todo

# todo
- it would be nice to treat states actually as coordinate pairs and not separate them in some places like probActions and rewards
"""
struct MDPConfig
    _probStates::Array{Float64, 5}
    _rewards::Array{Float64, 3}
    _nonBoundaryTerminalStates::Vector{Pair{Int}}
end

"""
    module MDP

Meant as quality of life functionality for functions like e.g. `transition!(mdp, ACTION_UP)`.

Currently unused and unimplemented.
"""
mutable struct MDP
    config::MDPConfig
    currentState::Tuple{Int, 2}
end


# todo: turn this into an enum
# right = 1, up = 2, left = 3, down = 4 (inspired from unit circle sine function stuff)
const ACTION_RIGHT = 1
const ACTION_UP = 2
const ACTION_LEFT = 3
const ACTION_DOWN = 4
const ACTIONS = [ACTION_RIGHT, ACTION_UP, ACTION_LEFT, ACTION_DOWN]

"""
    _assert_valid_state(mdp, state)

Asserts that state is included in the state space of mdp.
"""
function _assert_valid_state(mdp::MDPConfig, state::Pair{Int})
    dims = size(mdp._rewards)
    width = dims[1]
    height = dims[2]

    @assert 0 < state.first <= width
    @assert 0 < state.second <= height
end


"""
    _makeGridWorldTransitions(width, height)

Sets up state transitions for a simple grid world MDP.

Does not add a destination, i.e. a terminal state with positive reward. That is supposed to be done on the caller side because rewards have to altered accordingly and mismatches would mess up the entire simulation.

# Parameters
- `width::Int`: at least 3
- `height::Int`: at least 3
"""
function _makeGridworldTransitions(width::Int, height::Int)::Array{Float64, 5}
    @assert width >= 3
    @assert height >= 3

    probStates = zeros(width, height, length(ACTIONS), width, height)

    # set boundaries
    for x in 1:width
        probStates[x, 1, :, x, :] .= stdbasis(height, 1)'
        probStates[x, height, :, x, :] .= stdbasis(height, height)'
    end
    for y in 1:height
        probStates[1, y, :, :, y] .= stdbasis(width, 1)'
        probStates[width, y, :, :, y] .= stdbasis(width, width)'
    end

    # normal transitions
    for x in 2:width-1
        for y in 2:height-1
            probStates[x, y, ACTION_RIGHT, x+1, y] = 1.0
            probStates[x, y, ACTION_UP, x, y-1] = 1.0
            probStates[x, y, ACTION_LEFT, x-1, y] = 1.0
            probStates[x, y, ACTION_DOWN, x, y+1] = 1.0
        end
    end

    return probStates
end

"""
    _makeGridWorldRewards(width, height)

Sets up reward matrix r[x, y, action] for a simple grid world MDP.

# Parameters
- `width::Int`: at least 3
- `height::Int`: at least 3
"""
function _makeGridWorldRewards(width::Int, height::Int)::Array{Float64, 3}
    @assert width >= 3
    @assert height >= 3

    rewards = zeros(width, height, length(ACTIONS))

    # set boundaries
    for x in 2:width-1
        rewards[x, 2, ACTION_UP] = -10.0
        rewards[x, height-1, ACTION_DOWN] = -10.0
    end

    for y in 2:height-1
        rewards[2, y, ACTION_LEFT] = -10.0
        rewards[width-1, y, ACTION_RIGHT] = -10.0
    end

    return rewards
end

"""
    makeSimpleGridWorld(width, height)

Create MDP for simple gridworld consisting of a widthxheight grid. Boundaries are terminal states and transitioning to them yields negative rewards. The tile at (ceil(width*0.75), ceil(height*0.5)) is the destination, meaning a terminal state and yielding a positive reward when transitioning to it.

# Parameters
- `width::Int`: at least 3
- `height::Int`: at least 3
"""
function makeSimpleGridworld(width::Int, height::Int)::MDPConfig
    @assert width >= 3
    @assert height >= 3
    probStates = _makeGridworldTransitions(width, height)
    rewards = _makeGridWorldRewards(width, height)

    terminalStates = Pair{Int}[]

    # set destination
    # todo make sure that this doesn't land on the boundary
    xdest = ceil(Int, width*0.8)
    ydest = ceil(Int, height*0.5)

    probStates[xdest, ydest, :, :, :] = zeros(Float64, length(ACTIONS), width, height)
    probStates[xdest, ydest, :, xdest, ydest] = ones(Float64, length(ACTIONS))

    push!(terminalStates, Pair(xdest, ydest))

    rewards[xdest+1, ydest, ACTION_LEFT] = 1.0
    rewards[xdest, ydest+1, ACTION_UP] = 1.0
    rewards[xdest-1, ydest, ACTION_RIGHT] = 1.0
    rewards[xdest, ydest-1, ACTION_DOWN] = 1.0

    return MDPConfig(probStates, rewards, terminalStates)
end

"""
    transition(mdp, state, action)

Draw next state given state and action, and produce reward.

Returns `(xnew, ynew, reward)`.
"""
function transition(mdp::MDPConfig, state::Pair{Int}, action::Int)::Tuple{Pair{Int}, Float64}
    _assert_valid_state(mdp, state)
    @assert action in ACTIONS

    probabilities = mdp._probStates[state.first, state.second, action, :, :]
    distr = Categorical(vec(probabilities))
    sample = rand(distr)
    newState = Tuple(CartesianIndices(probabilities)[sample])
    @assert length(newState) == 2

    reward = mdp._rewards[state.first, state.second, action]

    return Pair(newState[1], newState[2]), reward
end

"""
    reward(mdp, state, action)

Return reward for state without having to transition. Mostly used for plotting.
"""
function reward(mdp::MDPConfig, state::Pair{Int}, action::Int)
    _assert_valid_state(mdp, state)

    return mdp._rewards[state.first, state.second, action]
end

"""
    _is_boundary(mdp, state)

Checks whether state lays on the boundary of mdp, i.e. on the outer edges.
"""
function _is_boundary(mdp::MDPConfig, state::Pair{Int})::Bool
    dims = size(mdp._rewards)
    width = dims[1]
    height = dims[2]

    return (state.first == 1 || state.first == width || state.second == 1 || state.second == height)
end

"""
    is_terminal(mdp, state)

Check if a grid world state is terminal.
"""
function is_terminal(mdp::MDPConfig, state::Pair{Int})::Bool
    _assert_valid_state(mdp, state)

    return _is_boundary(mdp, state) || (state in mdp._nonBoundaryTerminalStates)
end

function dimensions(mdp::MDPConfig)::Pair{Int}
    xdim, ydim, _ = size(mdp._rewards)

    return Pair(xdim, ydim)
end
end