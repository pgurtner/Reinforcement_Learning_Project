"""
    module Benchmark

The module containing everything around running benchmarks. Plotting is handled by the module Plots.

# ideas
- convergence
- performance compared to other approaches
- performance by itself??
- accuracy

3. implement benchmarking (something like figure 7.2, between algorithms for efficiency and by oneself for convergence, IMPORTANT: make sure not to mix up state-value and action-value functions, look again into how you can get the state-value function from the action-value function)

"""
module Benchmark


module nstepSarsa
    using CairoMakie
    using ...Learning
    using ...MDPs
    using ...GridWorldPlotting
    using ...Policies

    function basicConvergence()
        mdp = MDPs.makeSimpleGridworld(10, 10)
        π, Q, errors = Learning.nstepSarsa(mdp, n=10, ϵ=0.1, α=0.1, γ=0.5, nepisodes=10000)

        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            #yscale = log10,
            title="Basic convergence of n-step Sarsa"
        )

        lines!(ax, 1:length(errors), errors,
            linewidth=2,
          #  marker = :circle,
           # markersize = 8,
        )

        #axislegend(ax, position = :rt)
        display(fig)

        V = Policies.stateValueFunctionFromQ(π, Q)
        GridWorldPlotting.drawStateValueFunction(mdp, V)
    end

    function compareN()
    end

    function compareEps()
    end

    function compareAlpha()
    end
end


end