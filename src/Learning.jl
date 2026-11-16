"""
    module Learning

The module that contains all the learning algorithms for grid world MDPs. Currently only n-step Sarsa is implemented.
"""
module Learning
using ..MDPs
using ..Policies
using Random

"""
    nstepSarsa(mdp, n, ϵ, α, γ, nepisodes)

n-step Sarsa implementation for grid world MDPs. Overall code structure and variable names are kept as close to the book as possible.

# Parameters
- `mdp::MDPs.MDPConfig`: MDP for which a policy gets learned
- `n::Int`: step size
- `ϵ::Float64`: parameter for ϵ-greedy policy π
- `α::Float64`: learning rate
- `γ::Float64`: discount factor
- `nepisodes::Int`: number of episodes

# Returns
Learned policy π and learned action-value function Q.
"""
function nstepSarsa(mdp::MDPs.MDPConfig; n::Int, ϵ::Float64, α::Float64, γ::Float64, nepisodes::Int)::Tuple{Policies.Policy, Array{Float64, 3}}
    @assert n > 0
    @assert 0 <= ϵ <= 1
    @assert 0 <= α <= 1
    @assert 0 <= γ <= 1

    dims = MDPs.dimensions(mdp)

    # initialize action-value function to all zeros
    # States are usually handed around as integer pairs but stored in arrays, which often causes code like Q[state.first, state.second, action] to access Q values.
    Q = fill(0.0, dims.first, dims.second, length(MDPs.ACTIONS))

    # initialize π to a fixed all uniform policy
    π = Policies.makeUniformlyRandomPolicy(dims)

    # list of potential inital states, i.e. those that are not terminal
    initialStates = [Pair(x, y) for x in 1:dims[1] for y in 1:dims[2] if !MDPs.is_terminal(mdp, Pair(x, y))]

    # Julia array indices start from 1. To keep the algorithm close to the book, a +1 was added to all relevant array accesses.
    # The whitespace in e.g. S[t + 1 + 1] however has to be that way and can't be made more clear,
    # because in S[t+1 +1], Julia can't differentiate this from its matrix syntax leading to [t+1, +1]
    for _ in 1:nepisodes
        # list of encountered states
        S = Pair{Int}[]

        # randomly drawn initial state
        S1 = rand(initialStates)
        push!(S, S1)

        # list of encountered actions
        A = Int[]

        # draw first action from π
        A1 = Policies.evaluate(π, S[0 + 1])
        push!(A, A1)

        # list of encountered rewards
        R = Float64[]

        # R_0 doesn't really exist in the algorithm, just use as a placeholder to keep index accesses simple
        push!(R, 0.0)

        # Infinity is typically unavailable for integer type so it is implemented with a separate T_infinite boolean
        T = 0
        T_infinite = true
        τ = 0
        t = 0

        while T_infinite || τ < T - 1
            if T_infinite || t < T
                # take action and get next state and reward
                newS, nextR = MDPs.transition(mdp, S[t + 1], A[t + 1])
                push!(S, newS)
                push!(R, nextR)

                # if S_{t+1} is terminal
                if MDPs.is_terminal(mdp, S[t + 1 + 1])
                    T = t + 1
                    T_infinite = false
                else
                    # select and store A_{t+1}
                    push!(A, Policies.evaluate(π, S[t + 1 + 1]))
                end
            end
            
            τ = t - n + 1

            if τ >= 0
                # technical note: the calculation of G could be optimized by only subtracting R[τ] and adding R[τ+n], but then it would maybe be more unstable due to floating point stuff
                # also, n is rather small usually, so this shouldn't be much of an issue
                # for TD methods this would avoid quadratic runtime though

                # compute m = min(τ + n, T)
                m = if T_infinite
                    τ + n
                else
                    min(τ + n, T)
                end

                # the indexing is drawn into both factors γ and R
                # the exponent range for γ can be simplified moving the -τ-1 into the start and end points, i.e. (τ+1 : m) .- τ .- 1 = (0 : m-τ-1)
                # the .+ 1 for R is for moving the indices to Julia's indexing style
                G = sum((γ .^ ((0 : m-τ-1))) .* R[(τ+1 : m) .+ 1])
                if T_infinite || τ + n < T
                    # G += γ^n * Q(S_{τ+1}, A_{τ+n})
                    G += γ^n * Q[S[τ + n + 1].first, S[τ + n + 1].second, A[τ + n + 1]]
                end

                # the actual update to Q
                Q[S[τ + 1].first, S[τ + 1].second, A[τ + 1]] += α*(G - Q[S[τ + 1].first, S[τ + 1].second, A[τ + 1]])

                # updating π to be ϵ-greedy wrt. to Q
                Policies.updateState!(π, S[τ + 1], Q, ϵ)
            end
            
            t += 1
        end
    end

    return π, Q # todo: return deterministic π, not this ϵ-greedy one for learning
end

end