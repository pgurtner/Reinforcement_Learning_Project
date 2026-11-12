module Policies
using ..States

export Policy

struct Policy
    stateactionmatrix
end

function evaluate(π::Policy, s::GridState)
    # check state validness
    # setup distribution
    # draw action from distribution
end

end