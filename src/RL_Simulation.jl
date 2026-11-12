module RL_Simulation

include("States.jl")
include("Policies.jl")
include("Agents.jl")

function test()
    a::GridStateSpace = initial_state(GridStateSpace)
    b::TestCubeStateSpace = initial_state(TestCubeStateSpace)

    print(a)
    print(b)
end


end