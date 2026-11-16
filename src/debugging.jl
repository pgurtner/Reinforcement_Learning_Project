include("RL_Simulation.jl")

using .RL_Simulation

function test()
    println("biep")
    mdp = MDPs.GridWorld.makeSimpleGridworld(10, 10)

    π, Q = Learning.nstepSarsa(mdp, n=2, ϵ=0.1, α=0.1, γ=0.5, nepisodes=1000)
    @assert all(Q .<= 2.0) "with γ=0.5 and the highest reward of 1, Q can't surpass 2, because sum_0^inf 0.5^i = 2"
end

function testTransitions()
    for (i, action) in enumerate(MDPs.GridWorld.ACTIONS)
        newS, reward = MDPs.GridWorld.transition(mdp, S1, i)
        println("$i ($action): $newS, reward=$reward")
    end
end