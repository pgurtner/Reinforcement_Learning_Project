module Agents
using ..Policies

export Agent, OnPolicyAgent, OffPolicyAgent

# this abstract type is not really necessary because on/off policy is not a question about internal efficiency
# but conceptually it makes sense, maybe it'll be used at some point. If not, this is just an unused line of code
abstract type Agent end

struct OnPolicyAgent <: Agent
    π::Policy
end

struct OffPolicyAgent <: Agent
end

function _test()
    
end

end
