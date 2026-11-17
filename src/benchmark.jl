"""
    module Benchmark

The module containing everything around running benchmarks.
"""
module Benchmark

"""
    module nstepSarsa

Benchmarking for the nstepSarsa algorithm. Originally it was planned to also test the nstepQ(σ) algorithm, but as of now that was left out due to time reasons.
"""
module nstepSarsa
    using CairoMakie
    using ...Learning
    using ...MDPs
    using ...GridWorldPlotting
    using ...Policies

    """
        runBenchmark(mdp, n, ϵ, α, γ, nepisodes, repititions)
    
    Helper function for running n-step Sarsa with the given parameters for repetitions many times and averaging the recorded errors.
    """
    function runBenchmark(mdp::MDPs.MDPConfig; n::Int, ϵ::Float64, α::Float64, γ::Float64, nepisodes::Int, repititions::Int)::Tuple{Vector{Float64}, Vector{Float64}}
        VerrorsAvg = zeros(nepisodes)
        QerrorsAvg = zeros(nepisodes)

        for _ in 1:repititions
            _, _, Verrors, Qerrors = Learning.nstepSarsa(mdp; n, ϵ, α, γ, nepisodes)

            VerrorsAvg .+= Verrors
            QerrorsAvg .+= Qerrors
        end

        VerrorsAvg ./= repititions
        QerrorsAvg ./= repititions

        return VerrorsAvg, QerrorsAvg
    end

    """
        basicConvergence()
        
    Benchmark basic convergence, i.e. just one run without any varying parameters.
    """
    function basicConvergence()
        mdp = MDPs.makeSimpleGridworld(10, 10)

        Verror, Qerror = runBenchmark(mdp, n=1, ϵ=0.4, α=0.1, γ=0.5, nepisodes=10000, repititions=5)

        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="State-value function accuracy"
        )

        scatterlines!(ax, 1:length(Verror), Verror,
            linewidth=2,
            markersize=1,
        )
        
        save("out/plots/Verrors.png", fig)



        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="Action-value function accuracy"
        )

        scatterlines!(ax, 1:length(Qerror), Qerror,
            linewidth=2,
            markersize=1,
        )
        
        save("out/plots/Qerrors.png", fig)
    end

    """
        compareN()

    Benchmark n values from 1 to 10.
    """
    function compareN()
        mdp = MDPs.makeSimpleGridworld(10, 10)
        Verrors = []
        Qerrors = []
        N = 10

        for i in 1:N
            Verror, Qerror = runBenchmark(mdp, n=i, ϵ=0.4, α=0.1, γ=0.5, nepisodes=10000, repititions=5)
            push!(Verrors, Verror)
            push!(Qerrors, Qerror)
        end

        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="State-value function accuracy with varying n values"
        )

        colors = Makie.resample_cmap(:roma10, N)
        for i in 1:N
            scatterlines!(ax, 1:length(Verrors[i]), Verrors[i],
                linewidth=2,
                markersize=1,
                label="n=$i",
                color = colors[i]
            )
        end
        
        axislegend(ax, position = :rt)
        save("out/plots/compareN_Verrors.png", fig)



        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="Action-value function accuracy with varying n values"
        )

        colors = Makie.resample_cmap(:roma10, N)
        for i in 1:N
            scatterlines!(ax, 1:length(Qerrors[i]), Qerrors[i],
                linewidth=2,
                markersize=1,
                label="n=$i",
                color = colors[i]
            )
        end
        
        axislegend(ax, position = :rt)
        save("out/plots/compareN_Qerrors.png", fig)

    end

    """
        compareEps()
    
    Benchmark ϵ values from 0.1 to 0.9.
    """
    function compareEps()
        mdp = MDPs.makeSimpleGridworld(10, 10)
        Verrors = []
        Qerrors = []

        for i in 1:9
            Verror, Qerror = runBenchmark(mdp, n=1, ϵ=0.1*i, α=0.1, γ=0.5, nepisodes=10000, repititions=5)
            push!(Verrors, Verror)
            push!(Qerrors, Qerror)
        end

        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="State-value function accuracy with varying ϵ values"
        )

        colors = Makie.resample_cmap(:roma10, 10)
        for i in 1:9
            scatterlines!(ax, 1:length(Verrors[i]), Verrors[i],
                linewidth=2,
                markersize=1,
                label="ϵ=$(round(0.1*i, digits=1))",
                color=colors[i]
            )
        end
        
        axislegend(ax, position = :rt)
        save("out/plots/compareEps_Verrors.png", fig)


        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="Action-value function accuracy with varying ϵ values"
        )

        colors = Makie.resample_cmap(:roma10, 10)
        for i in 1:9
            scatterlines!(ax, 1:length(Qerrors[i]), Qerrors[i],
                linewidth=2,
                markersize=1,
                label="ϵ=$(round(0.1*i, digits=1))",
                color=colors[i]
            )
        end
        
        axislegend(ax, position = :rt)
        save("out/plots/compareEps_Qerrors.png", fig)

    end

    """
        compareAlpha

    Benchmark α values from 0.1 to 1.0.
    """
    function compareAlpha()
        mdp = MDPs.makeSimpleGridworld(10, 10)
        Verrors = []
        Qerrors = []

        A = 7
        for i in 1:10
            Verror, Qerror = runBenchmark(mdp, n=1, ϵ=0.4, α=0.1*i, γ=0.5, nepisodes=10000, repititions=5)
            push!(Verrors, Verror)
            push!(Qerrors, Qerror)
        end

        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="State-value function accuracy with varying α values"
        )

        # colors = Makie.resample_cmap(:roma10, 10)
        for i in 1:3
            scatterlines!(ax, 1:length(Verrors[i]), Verrors[i],
                linewidth=2,
                markersize=1,
                label="α=$(round(i*0.1, digits=1))",
               # color = colors[i]
            )
        end
        
        axislegend(ax, position = :rt)
        save("out/plots/compareAlpha_Verrors.png", fig)


        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1],
            xlabel = "Episode",
            ylabel="Mean squared error",
            yscale = log10,
            title="Action-value function accuracy with varying α values"
        )

        colors = Makie.resample_cmap(:roma10, 10)
        for i in 1:A
            scatterlines!(ax, 1:length(Qerrors[i]), Qerrors[i],
                linewidth=2,
                markersize=1,
                label="α=$(round(i*0.1, digits=1))",
                color = colors[i]
            )
        end
        
        axislegend(ax, position = :rt)
        save("out/plots/compareAlpha_Qerrors.png", fig)
    end

    function runAll()
        println("running basic convergence benchmarks...")
        basicConvergence()

        println("running n value comparison benchmarks...")
        compareN()

        println("running ϵ value comparison benchmarks...")
        compareEps()

        println("running α value comparison benchmarks...")
        compareAlpha()
    end
end


end