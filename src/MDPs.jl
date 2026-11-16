"""
    module MDPs

Module that implements the interface for grid world Markov Decision Processes.

# todo
- the module name is bad
"""
module MDPs
using Random
using Distributions
using ...Utils: stdbasis

"""
    MDPConfig(_probActions, _rewards, _nonBoundaryTerminalStates)

Configuration of a grid world Markov Decision process where the grid edges are always terminal states.

This struct is meant to be instantiated from this module and only handed to functions of this module without accessing its fields from the caller side.

# Fields
- `_probStates::Array{Float64, 5}`: extremely sparse array that gives the probability to move from one state to another, could be stored as a sparse array type
- `_rewards::Array{Float64, 3}`: reward per state
- `_nonBoundaryTerminalStates::Vector{Pair{Int}}`: terminal states that aren't already boundaries
"""
struct MDPConfig
    _probStates::Array{Float64, 5}
    _rewards::Array{Float64, 3}
    _nonBoundaryTerminalStates::Vector{Pair{Int}}
end

"""
    module MDP

Meant as quality of life functionality for functions like e.g. `transition!(mdp, ACTION_UP)` to avoid having to hand around the current state.

Currently unused and unimplemented.
"""
mutable struct MDP
    config::MDPConfig
    currentState::Tuple{Int, 2}
end

# set of actions used for the grid world MDPs
# todo: turn this into an enum
# right = 1, up = 2, left = 3, down = 4 (inspired from unit circle sine function stuff)
const ACTION_RIGHT = 1
const ACTION_UP = 2
const ACTION_LEFT = 3
const ACTION_DOWN = 4
const ACTIONS = [ACTION_RIGHT, ACTION_UP, ACTION_LEFT, ACTION_DOWN]

"""
    _assert_valid_state(mdp, state)

Checks that state is included in the state space of mdp.
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

Does not add a destination, i.e. a terminal state with positive reward. That is supposed to be done on the caller side because rewards have to be altered accordingly and mismatches would mess up the entire simulation.

# Parameters
- `width::Int`: width of grid world
- `height::Int`: height of grid world

# Returns
State transition probabilites meant to be used for MDPConfig._probStates.
"""
function _makeGridworldTransitions(width::Int, height::Int)::Array{Float64, 5}
    # initialize to all zero
    probStates = zeros(width, height, length(ACTIONS), width, height)

    # set boundaries
    # probStates[a, b, action, c, d] is the probability to get from (a, b) to (c, d) when taking action
    # here, boundaries are terminal states, so set all probabilities to 1.0 for the respective state
    # this is done by constructing (1.0, 0.0, 0.0, ..., 0.0) or (0.0, 0.0, ..., 1.0) with stdbasis(length, index)
    for x in 1:width
        # v' is Julia syntax for the transpose of v
        # the first dimension of probStates[x, 1, :, x, :] is along the actions axis so the (1.0, 0.0, 0.0, ...) vectors have to transposed
        probStates[x, 1, :, x, :] .= stdbasis(height, 1)'
        probStates[x, height, :, x, :] .= stdbasis(height, height)'
    end
    for y in 1:height
        probStates[1, y, :, :, y] .= stdbasis(width, 1)'
        probStates[width, y, :, :, y] .= stdbasis(width, width)'
    end

    # normal transitions
    # this simple grid world example uses deterministic state transitions, so only set the probability for the state in the respective direction to 1.0 and leave the rest at 0.0
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

Sets up reward matrix r[x, y, action] for a simple grid world MDP while leaving out any destination with positive reward.

Terminal states with positive rewards are set on the caller side because the state transitions have to be altered accordingly.

# Parameters
- `width::Int`: width of grid world
- `height::Int`: height of grid world

# Returns
Rewards per state meant for MDPConfig._rewards.
"""
function _makeGridWorldRewards(width::Int, height::Int)::Array{Float64, 3}
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

Create MDP for simple a grid world consisting of a width x height grid.
    
Boundaries are terminal states and transitioning to them yields negative rewards.

The tile at (ceil(width*0.8), ceil(height*0.5)) is the destination, meaning a terminal state and yielding a positive reward when transitioning to it.

# Parameters
- `width::Int`: width of grid world
- `height::Int`: height of grid world

# Returns
MDP configuration resembling a simple grid world with boundaries and one destination state.
"""
function makeSimpleGridworld(width::Int, height::Int)::MDPConfig
    # todo: actually the algorithms need at least one non-terminal state
    # todo: also make sure that it's large enough so the destination state doesn't land on the boundary
    @assert width >= 3
    @assert height >= 3

    probStates = _makeGridworldTransitions(width, height)
    rewards = _makeGridWorldRewards(width, height)

    terminalStates = Pair{Int}[]

    # set destination
    xdest = ceil(Int, width*0.8)
    ydest = ceil(Int, height*0.5)

    # limit transitions from destination state to stay on itself, i.e. make it terminal
    probStates[xdest, ydest, :, :, :] = zeros(Float64, length(ACTIONS), width, height)
    probStates[xdest, ydest, :, xdest, ydest] = ones(Float64, length(ACTIONS))

    push!(terminalStates, Pair(xdest, ydest))

    # set positive rewards to transition onto the destination state
    rewards[xdest+1, ydest, ACTION_LEFT] = 1.0
    rewards[xdest, ydest+1, ACTION_UP] = 1.0
    rewards[xdest-1, ydest, ACTION_RIGHT] = 1.0
    rewards[xdest, ydest-1, ACTION_DOWN] = 1.0

    return MDPConfig(probStates, rewards, terminalStates)
end

"""
    transition(mdp, state, action)

Draw next state given state and action, and produce reward.

# Parameters
- `mdp::MDPConfig`: respective MDP
- `state::Pair{Int}`: current state
- `action::Int`: respective action

# Returns
Drawn state and reward.
"""
function transition(mdp::MDPConfig, state::Pair{Int}, action::Int)::Tuple{Pair{Int}, Float64}
    _assert_valid_state(mdp, state)
    @assert action in ACTIONS

    # get transition probabilities for state and action
    probabilities = mdp._probStates[state.first, state.second, action, :, :]

    # turn them into a distribution
    # the vec call turns the matrix into a linear vector, which is needed by the Categorical constructor
    distr = Categorical(vec(probabilities))

    # draw an action
    sample = rand(distr)

    # get from the linear vector representation back to the 2d matrix indices
    newState = Tuple(CartesianIndices(probabilities)[sample])
    @assert length(newState) == 2

    # pick reward, this implementation uses deterministic rewards
    reward = mdp._rewards[state.first, state.second, action]

    return Pair(newState[1], newState[2]), reward
end

"""
    reward(mdp, state, action)

Return reward for state without having to transition. Only used for plotting, not learning.

# Parameters
- `mdp::MDPConfig`: respective MDP
- `state::Pair{Int}`: state from which action is 'taken' (without actually transitioning)
- `action::Int`: 'taken' action

# Returns
Respective reward value.
"""
function reward(mdp::MDPConfig, state::Pair{Int}, action::Int)::Float64
    _assert_valid_state(mdp, state)
    @assert action in ACTIONS

    return mdp._rewards[state.first, state.second, action]
end

"""
    _is_boundary(mdp, state)

Checks whether state lays on the boundary of mdp, i.e. on the outer edges.

# Parameters
- `mdp::MDPConfig`: respective MDP
- `state::Pair{Int}`: state that gets checked
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

# Parameters
- `mdp::MDPConfig`: respective MDP
- `state::Pair{Int}`: state that gets checked
"""
function is_terminal(mdp::MDPConfig, state::Pair{Int})::Bool
    _assert_valid_state(mdp, state)

    return _is_boundary(mdp, state) || (state in mdp._nonBoundaryTerminalStates)
end

"""
    dimensions(mdp)

Getter function for spatial dimensions of mdp to avoid having to access hidden fields of MDPConfig that aren't meant to be directly accessed outside this module.

# Parameters
- `mdp::MDPConfig`: respective MDP

# Returns
Width and height of the grid world of mdp, i.e. also the first two dimensions of MDPConfig._probStates and MDPConfig._rewards.
"""
function dimensions(mdp::MDPConfig)::Pair{Int}
    xdim, ydim, _ = size(mdp._rewards)

    return Pair(xdim, ydim)
end
end