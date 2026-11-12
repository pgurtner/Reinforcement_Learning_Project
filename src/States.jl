module States

export GridState, GridStateSpace, TestCubeStateSpace, initial_state

# todo: somehow integrate abstract type State ??

struct GridState
    x::Int
    y::Int
end

struct GridStateSpace
    width::Int
    height::Int
end

struct TestCubeStateSpace
    width::Int
    height::Int
    depth::Int
end

function initial_state(::GridStateSpace)
    return GridState(0, 0)
end

function initial_state(::TestCubeStateSpace)
    return TestCubeStateSpace(0, 0, 0)
end

end