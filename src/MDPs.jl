module MDPs

# todo: ideally this could use some abstract MDP functionality while GridWorld is only one implementation

# todo grid world matrices are extremely sparce, a sparse matrix type would be more efficient
module GridWorld
    using Random
    using Distributions
    using ...Utils: stdbasis

    struct MDPConfig
        probActions::Array{Float64, 5}
        rewards::Array{Float64, 3}
    end

    # todo currently unused
    mutable struct MDP
        config::MDPConfig
        currentState::Tuple{Int, 2}
    end


    # right = 1, up = 2, left = 3, down = 4 (inspired from unit circle sine function stuff)
    const ACTION_RIGHT = 1
    const ACTION_UP = 2
    const ACTION_LEFT = 3
    const ACTION_DOWN = 4
    const ACTIONS = [ACTION_RIGHT, ACTION_UP, ACTION_LEFT, ACTION_DOWN]

    # doesn't set destination, that should be done together with setting rewards to avoid mismatches
    function _makeGridworldTransitions(width::Int, height::Int)
        @assert width >= 3
        @assert height >= 3

        probActions = zeros(width, height, length(ACTIONS), width, height)

        # set boundaries
        for x in 1:width
            probActions[x, 1, :, x, :] .= stdbasis(height, 1)'
            probActions[x, end, :, x, :] .= stdbasis(height, height)'
        end
        for y in 1:height
            probActions[1, y, :, :, y] .= stdbasis(width, 1)'
            probActions[end, y, :, :, y] .= stdbasis(width, width)'
        end

        # normal transitions
        for x in 2:width-1
            for y in 2:height-1
                probActions[x, y, ACTION_RIGHT, x+1, y] = 1.0
                probActions[x, y, ACTION_UP, x, y-1] = 1.0
                probActions[x, y, ACTION_LEFT, x-1, y] = 1.0
                probActions[x, y, ACTION_DOWN, x, y+1] = 1.0
            end
        end

        return probActions
    end

    function _makeGridWorldRewards(width::Int, height::Int)
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

    function makeSimpleGridworld(width::Int, height::Int)
        probActions = _makeGridworldTransitions(width, height)
        rewards = _makeGridWorldRewards(width, height)

        # set destination
        xdest = ceil(Int, width*0.75)
        ydest = ceil(Int, height*0.5)

        probActions[xdest, ydest, :, :, :] = zeros(Float64, length(ACTIONS), width, height)
        probActions[xdest, ydest, :, xdest, ydest] = ones(Float64, length(ACTIONS))

        rewards[xdest+1, ydest, ACTION_LEFT] = 1.0
        rewards[xdest, ydest+1, ACTION_UP] = 1.0
        rewards[xdest-1, ydest, ACTION_RIGHT] = 1.0
        rewards[xdest, ydest-1, ACTION_DOWN] = 1.0

        return MDPConfig(probActions, rewards)
    end

    function transition(mdp::MDPConfig, xstate::Int, ystate::Int, action::Int)
        dims = size(mdp.rewards)
        @assert xstate <= dims[1]
        @assert xstate > 0
        @assert ystate <= dims[2]
        @assert ystate > 0
        @assert action in ACTIONS

        probabilities = mdp.probActions[xstate, ystate, action, :, :]
        distr = Categorical(vec(probabilities))
        sample = rand(distr)
        xnew, ynew = Tuple(CartesianIndices(probabilities)[sample])

        reward = mdp.rewards[xstate, ystate, action]

        return xnew, ynew, reward
    end

end

end