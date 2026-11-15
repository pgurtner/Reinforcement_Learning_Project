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
    terminalStates = [(x, y) for x in 1:width for y in 1:height if MDPs.GridWorld.is_terminal(mdp, Pair(x ,y))]
    scatter!(ax, first.(terminalStates), last.(terminalStates);
        marker=:circle,
        markersize=25,
        color=:black
    )

    # todo: don't hardcode this but turn it into a parameter or something
    xstart = ceil(Int, dims.first*0.2)
    ystart = ceil(Int, dims.second*0.5)
    scatter!(ax, [xstart], [ystart]; marker=:circle, markersize=25, color=:green)

    return fig, ax
end

function drawQ(Q::Array{Float64, 3})
end

# currently only draws arrows when argmax action has a non-zero value to avoid displaying unlearnt actions
function drawOptimalActions(mdp::MDPs.GridWorld.MDPConfig, Q::Array{Float64, 3})
    # todo: check that dimensions of mdp and Q match

    fig, ax = drawGridWorld(mdp)

    width, height = MDPs.GridWorld.dimensions(mdp)
    qdims = size(Q)

    for x in 1:width
        for y in 1:height
            if MDPs.GridWorld.is_terminal(mdp, Pair(x, y))
                continue
            end

            # action = argmax(Q[x, y, :])
            # if Q[x, y, action] == 0.0
            #     continue
            # end

            # dx = 0
            # dy = 0
            # if action == MDPs.GridWorld.ACTION_RIGHT
            #     dx = 1
            # elseif action == MDPs.GridWorld.ACTION_UP
            #     dy = -1
            # elseif action == MDPs.GridWorld.ACTION_LEFT
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
            arrows2d!(ax, [x], [y], [1], [0], color = c(Q[x, y, MDPs.GridWorld.ACTION_RIGHT]), lengthscale=abs(Q[x, y, MDPs.GridWorld.ACTION_RIGHT])*scale)
            arrows2d!(ax, [x], [y], [0], [-1], color = c(Q[x, y, MDPs.GridWorld.ACTION_UP]), lengthscale=abs(Q[x, y, MDPs.GridWorld.ACTION_UP])*scale)
            arrows2d!(ax, [x], [y], [-1], [0], color = c(Q[x, y, MDPs.GridWorld.ACTION_LEFT]), lengthscale=abs(Q[x, y, MDPs.GridWorld.ACTION_LEFT])*scale)
            arrows2d!(ax, [x], [y], [0], [1], color = c(Q[x, y, MDPs.GridWorld.ACTION_DOWN]), lengthscale=abs(Q[x, y, MDPs.GridWorld.ACTION_DOWN])*scale)
        end
    end

    display(fig)
end

end