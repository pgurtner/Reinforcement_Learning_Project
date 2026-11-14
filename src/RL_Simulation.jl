module RL_Simulation

include("utils.jl")

include("MDPs.jl")

include("Policies.jl")

include("Learning.jl")

function test()
    a::GridStateSpace = initial_state(GridStateSpace)
    b::TestCubeStateSpace = initial_state(TestCubeStateSpace)

    print(a)
    print(b)
end


end