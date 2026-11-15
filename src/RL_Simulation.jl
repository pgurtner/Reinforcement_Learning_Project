module RL_Simulation
include("plot.jl")
include("utils.jl")

include("MDPs.jl")

include("Policies.jl")

include("Learning.jl")

using .MDPs
using .Learning

function test()
    println("biep")
    mdp = MDPs.GridWorld.makeSimpleGridworld(10, 10)

    π, Q = Learning.nstepSarsa(mdp, n=1, ϵ=0.1, α=0.9, γ=1.0, nepisodes=1000)
end

end