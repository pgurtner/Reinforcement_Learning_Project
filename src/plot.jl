"""
    module Plot

Functions for plotting graphs and visualizing the grid worlds and things like action-value functions or state-value functions.
"""
module Plot
using CairoMakie
using ..MDPs

"""
    setupGridWorldGraphic(mdp)

Draw grid world of mdp as a plain grid and mark terminal states.

This conveys little new information and is mostly used as a helper for the other drawing functions.

# Parameters
- `mdp::MDPs.MDPConfig`: MDP that gets drawn

# Returns
Figure and axis instances for further drawing or displaying/saving.
"""
function setupGridWorldGraphic(mdp::MDPs.MDPConfig)::Tuple{Makie.Figure, Makie.Axis}
    dims = MDPs.dimensions(mdp)
    width = dims[1]
    height = dims[2]

    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1],
        aspect = DataAspect(),
        yreversed = true,
        xaxisposition = :top,
        xgridvisible = false,
        ygridvisible = false,
        xticks = 0:width,
        yticks = 0:height,
    )

    # draw grid
    # has to be drawn like this in loops because it appearantly doesn't use coordinate lists to summarize several drawing calls but to create polylines
    for i in 0:width + 1
        lines!(ax, [i, i] .+ 0.5, [0, height] .+ 0.5, color = :black)
    end

    for j in 0:height + 1
        lines!(ax, [0, width] .+ 0.5, [j, j] .+ 0.5, color = :black)
    end

    # limit x and y axes
    xlims!(ax, 0, width + 1)
    ylims!(ax, height + 1, 0)

    # draw terminal states
    terminalStates = [(x, y) for x in 1:width for y in 1:height if MDPs.is_terminal(mdp, Pair(x ,y))]
    scatter!(ax, first.(terminalStates), last.(terminalStates);
        marker=:circle,
        markersize=25,
        color=:black
    )

    return fig, ax
end

# currently only draws arrows when argmax action has a non-zero value to avoid displaying unlearnt actions
"""
    drawQ(mdp, Q)

Visualizes the action-value function Q by drawing each action value per state as an arrow in the respective direction.

The arrow lengths indicate the absolute value and they are normalized per state, such that the highest absolute value results in an arrow of length 0.5 (to avoid clutter across grid edges).

The signs of the values are visualized with colors: blue means positive, red is negative.

# Parameters
- `mdp::MDPs.MDPConfig`: respective MDP
- `Q::Array{Float64, 3}`: action-value function that gets drawn
"""
function drawQ(mdp::MDPs.MDPConfig, Q::Array{Float64, 3})
    # todo: check that dimensions of mdp and Q match

    fig, ax = setupGridWorldGraphic(mdp)

    width, height = MDPs.dimensions(mdp)
    for x in 1:width
        for y in 1:height
            # ignore terminal states
            if MDPs.is_terminal(mdp, Pair(x, y))
                continue
            end

            # normalize arrow lengths per state
            scale = 0.5 / maximum(abs.(Q[x, y, :]))

            # color depending on the sign of the value
            c(v) = if v >= 0
                :blue
            else
                :red
            end

            # arrows2d! uses coordinates and direction for drawing, not another pair of coordinates
            arrows2d!(ax, [x], [y], [1], [0], color = c(Q[x, y, MDPs.ACTION_RIGHT]), lengthscale=abs(Q[x, y, MDPs.ACTION_RIGHT])*scale)
            arrows2d!(ax, [x], [y], [0], [-1], color = c(Q[x, y, MDPs.ACTION_UP]), lengthscale=abs(Q[x, y, MDPs.ACTION_UP])*scale)
            arrows2d!(ax, [x], [y], [-1], [0], color = c(Q[x, y, MDPs.ACTION_LEFT]), lengthscale=abs(Q[x, y, MDPs.ACTION_LEFT])*scale)
            arrows2d!(ax, [x], [y], [0], [1], color = c(Q[x, y, MDPs.ACTION_DOWN]), lengthscale=abs(Q[x, y, MDPs.ACTION_DOWN])*scale)
        end
    end

    display(fig)
end

"""
    drawOptimalActions(mdp, Q)

Visualize each optimal actions per state. Ties are broken by whatever logic Julia's argmax function uses. 

# Parameters
- `mdp::MDPs.MDPConfig`: respective MDP
- `Q::Array{Float64, 3}`: action-value function that gets drawn
"""
function drawOptimalActions(mdp::MDPs.MDPConfig, Q::Array{Float64, 3})
    # todo: check that dimensions of mdp and Q match
    fig, ax = setupGridWorldGraphic(mdp)

    width, height = MDPs.dimensions(mdp)

    for x in 1:width
        for y in 1:height
            # skip terminal states
            if MDPs.is_terminal(mdp, Pair(x, y))
                continue
            end

            # get optimal action
            action = argmax(Q[x, y, :])

            # skip states that haven't learnt a positive value yet
            # only relevant when looking at first few algorithm episodes
            if Q[x, y, action] == 0.0
                continue
            end

            # direction of the respective arrow
            dx = 0
            dy = 0
            if action == MDPs.ACTION_RIGHT
                dx = 1
            elseif action == MDPs.ACTION_UP
                dy = -1
            elseif action == MDPs.ACTION_LEFT
                dx = -1
            else
                dy = 1
            end

            arrows2d!(ax,
                [x], [y],
                [dx], [dy],
                color = :black,
                lengthscale = 0.5
            )
        end
    end

    display(fig)
end

"""
    drawStateValueFunction(mdp, V)

Visualizes the state-value function V by displaying it as a heatmap and writing the individual values rounded to one decimal.

# Parameters
- `mdp::MDPs.MDPConfig`: respective MDP
- `V::Array{Float64, 2}`: state-value function to be visualized
"""
function drawStateValueFunction(mdp::MDPs.MDPConfig, V::Array{Float64, 2})
    fig, ax = setupGridWorldGraphic(mdp)

    width, height = MDPs.dimensions(mdp)

    # draws a heatmap, i.e. a grid where each tile gets a color from a spectrum based on the respective numerical value in V
    heatmap!(ax, 1:width, 1:height, V)

    # write each entry in V on top of the heatmap
    for x in 1:width
        for y in 1:height
            # skip terminal states
            if MDPs.is_terminal(mdp, Pair(x, y))
                continue
            end

            text!(ax, x, y, text=string(round(V[x, y], digits=1)), align=(:center, :center))
        end
    end

    display(fig)
end

end