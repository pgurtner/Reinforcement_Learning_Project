module Policies

export Policy

struct Policy
    stateactionmatrix
end

function evaluate(π::Policy)
    # check state validness
    # setup distribution
    # draw action from distribution
end

end