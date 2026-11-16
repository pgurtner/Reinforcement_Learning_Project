"""
    module Plot

# ideas
- plot plain grid world
- plot state value function
- plot best action per state (basically argmax on Q)
- plot sequence of taken routes
"""
module Plot
using CairoMakie
using ..MDPs

function drawGridWorld(mdp::MDPs.MDPConfig)
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
    # todo: draw this in one go, not in loops
    for i in 0:width + 1
        lines!(ax, [i, i] .+ 0.5, [0, height] .+ 0.5, color = :black)
    end

    for j in 0:height + 1
        lines!(ax, [0, width] .+ 0.5, [j, j] .+ 0.5, color = :black)
    end

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

function drawQ(Q::Array{Float64, 3})
end

# currently only draws arrows when argmax action has a non-zero value to avoid displaying unlearnt actions
function drawOptimalActions(mdp::MDPs.MDPConfig, Q::Array{Float64, 3})
    # todo: check that dimensions of mdp and Q match

    fig, ax = drawGridWorld(mdp)

    width, height = MDPs.dimensions(mdp)
    qdims = size(Q)

    for x in 1:width
        for y in 1:height
            if MDPs.is_terminal(mdp, Pair(x, y))
                continue
            end

            # action = argmax(Q[x, y, :])
            # if Q[x, y, action] == 0.0
            #     continue
            # end

            # dx = 0
            # dy = 0
            # if action == MDPs.ACTION_RIGHT
            #     dx = 1
            # elseif action == MDPs.ACTION_UP
            #     dy = -1
            # elseif action == MDPs.ACTION_LEFT
            #     dx = -1
            # else
            #     dy = 1
            # end

            # arrows2d!(ax,
            #     [x], [y],
            #     [dx], [dy],
            #     color = :blue,
            #     lengthscale = 0.5
            # )

            # max * scale = 0.5
            scale = 0.5 / maximum(abs.(Q[x, y, :]))
            c(v) = if v >= 0
                :blue
            else
                :red
            end
            arrows2d!(ax, [x], [y], [1], [0], color = c(Q[x, y, MDPs.ACTION_RIGHT]), lengthscale=abs(Q[x, y, MDPs.ACTION_RIGHT])*scale)
            arrows2d!(ax, [x], [y], [0], [-1], color = c(Q[x, y, MDPs.ACTION_UP]), lengthscale=abs(Q[x, y, MDPs.ACTION_UP])*scale)
            arrows2d!(ax, [x], [y], [-1], [0], color = c(Q[x, y, MDPs.ACTION_LEFT]), lengthscale=abs(Q[x, y, MDPs.ACTION_LEFT])*scale)
            arrows2d!(ax, [x], [y], [0], [1], color = c(Q[x, y, MDPs.ACTION_DOWN]), lengthscale=abs(Q[x, y, MDPs.ACTION_DOWN])*scale)
        end
    end

    display(fig)
end

function drawDeterministicPolicy(mdp::MDPs.MDPConfig, Q::Array{Float64, 3})
    # todo: check that dimensions of mdp and Q match
    fig, ax = drawGridWorld(mdp)

    width, height = MDPs.dimensions(mdp)

    for x in 1:width
        for y in 1:height
            if MDPs.is_terminal(mdp, Pair(x, y))
                continue
            end

            action = argmax(Q[x, y, :])
            if Q[x, y, action] == 0.0
                continue
            end

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

function drawStateValueFunction(mdp::MDPs.MDPConfig, V::Array{Float64, 2})
    fig, ax = drawGridWorld(mdp)

    width, height = MDPs.dimensions(mdp)

    heatmap!(ax, 1:width, 1:height, V)

    # text!(ax, 1:width, 1:height, text=string.(V), align=(:center, :center))
    for x in 1:width
        for y in 1:height
            if MDPs.is_terminal(mdp, Pair(x, y))
                continue
            end

            text!(ax, x, y, text=string(round(V[x, y], digits=1)), align=(:center, :center))
        end
    end
    
    display(fig)
end

end