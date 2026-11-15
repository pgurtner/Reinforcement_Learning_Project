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

function drawGridWorld(mdp::MDPs.GridWorld.MDPConfig)
    dims = MDPs.GridWorld.dimensions(mdp)
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
    for i in 0:width + 1
        lines!(ax, [i, i] .+ 0.5, [0, height] .+ 0.5, color = :black)
    end

    for j in 0:height + 1
        lines!(ax, [0, width] .+ 0.5, [j, j] .+ 0.5, color = :black)
    end

    xlims!(ax, 0, width + 1)
    ylims!(ax, height + 1, 0)

    # draw terminal states
    # todo: this loop can probably be turned into one scatter call
    for x in 1:width
        for y in 1:height
            if MDPs.GridWorld.is_terminal(mdp, Pair(x, y))
                scatter!(ax, [x], [y];
                    marker=:circle,
                    markersize=25,
                    color=:black
                )
            end
        end
    end


    return fig, ax
end

function drawQ(Q::Array{Float64, 3})
end

function drawOptimalActions(mdp::MDPs.GridWorld.MDPConfig, Q::Array{Float64, 3})
    # todo: check that dimensions of mdp and Q match

    fig, ax = drawGridWorld(mdp)

    width, height = MDPs.GridWorld.dimensions(mdp)

    for x in 1:width
        for y in 1:height
            if MDPs.GridWorld.is_terminal(mdp, Pair(x, y))
                continue
            end

            action = argmax(Q[x, y, :])
            dx = 0
            dy = 0
            if action == MDPs.GridWorld.ACTION_RIGHT
                dx = 1
            elseif action == MDPs.GridWorld.ACTION_UP
                dy = -1
            elseif action == MDPs.GridWorld.ACTION_LEFT
                dx = -1
            else
                dy = 1
            end

            arrows2d!(ax,
                [x], [y],
                [dx], [dy],
                color = :blue,
                lengthscale = 0.5
            )
        end
    end

    display(fig)
end

end