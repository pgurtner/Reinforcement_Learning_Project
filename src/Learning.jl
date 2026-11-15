module Learning
using ..MDPs
using ..Policies

export nstepSarsa

"""
    nstepSarsa(mdp, n, ϵ, α)

n-step Sarsa implementation for grid world MDPs.

# Parameters
- `mdp::GridWorld.MDPConfig`: MDP for which a policy gets learned
- `n::Int`: step size
- `ϵ::Float64`: parameter for ϵ-greedy policy π
- `α::Float64`: learning rate
"""
function nstepSarsa(mdp::MDPs.GridWorld.MDPConfig; n::Int, ϵ::Float64, α::Float64, γ::Float64, nepisodes::Int)::Tuple{Policies.GridWorld.Policy, Array{Float64, 3}}
    @assert n > 0
    @assert 0 < ϵ <= 1 # todo are these bounds correct?
    @assert 0 < α <= 1

    dims = MDPs.GridWorld.dimensions(mdp)

    Q = fill(0.0, dims.first, dims.second, length(MDPs.GridWorld.ACTIONS))
    π = Policies.GridWorld.makeUniformlyRandomPolicy(dims)
    S1 = Pair(ceil(Int, dims.first*0.2), ceil(Int, dims.second*0.5))

    for _ in 1:nepisodes
        S = Pair{Int}[]
        push!(S, S1)

        A = Int[]
        push!(A, Policies.GridWorld.evaluate(π, S[1]))

        R = Float64[]
        # R_0 doesn't really exist in the algorithm, just use 0.0
        push!(R, 0.0)

        T = 0
        T_infinite = true
        τ = 0
        t = 1

        while T_infinite || τ < T - 1
            if T_infinite || t < T
                newS, nextR = MDPs.GridWorld.transition(mdp, S[t], A[t])
                push!(S, newS)
                push!(R, nextR)

                if MDPs.GridWorld.is_terminal(mdp, S[t+1])
                    T = t + 1
                    T_infinite = false
                else
                    push!(A, Policies.GridWorld.evaluate(π, S[t+1]))
                end
            end
            
            τ = t - n + 1
            
            if τ >= 0
                # the calculation of G could be optimized by only subtracting R[τ] and adding R[τ+n], but then it would maybe be more unstable due to floating point stuff
                # also, n is rather small usually, so this shouldn't be much of an issue
                # for TD methods this would avoid quadratic runtime though
                l = if T_infinite
                    τ + n
                else
                    min(τ + n, T)
                end

                G = sum((γ .^ ((0 : l-τ-1))) .* R[τ+1 : l])
                if T_infinite || τ + n < T
                    # todo: check if the A[τ+n] is actually valid here
                    G += γ^n * Q[S[τ+n].first, S[τ+n].second, A[τ+n]]
                end

                Q[S[τ].first, S[τ].second, A[τ]] += α*(G - Q[S[τ].first, S[τ].second, A[τ]])

                Policies.GridWorld.updateState!(π, S[τ], Q, ϵ)
            end
            
            t += 1
        end
    end

    return π, Q # todo: return deterministic π, not this ϵ-greedy one for learning
end

end