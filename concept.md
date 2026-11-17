Note, some of the plots are different from the report because I did some extra tests while trying to understand the benchmarks that were included in the report.

# task
 * choose a method from chapter 7
 * describe it
 * implement it in a freely chosen example
 * "explore" the method by running the code (probably some experiments)
 * code has to be in the appendix
 * report has to explain subfunctions of the code
 * code has to have comments
 * report has to reference code comments

# roadmap

Multiple iterations from basic to interesting:
1. basic n-step Sarsa for typical grid-world
2. n-step Q(sigma) for typical grid-world
3. more complex grid-world
4. hide and seek where both hider and seeker learn and states can't just be grid-tiles

# todo
1. just implement gridworld example while keeping interfaces flexible
    4. implement nstepQ(\sigma)
    12. generate docs for code
* roadmap
* timeplan including report (5 days including today/thursday and excluding tuesday)
* find better names for modules

# code concept (very outdated)
 * Algorithm
 * MDP
 * example / problem
 * rest of the code is independent of the individual problem

Agent:
    plays according to learning algorithm and afterwards according to learned policy
    each action interacts with the environment -> or more generally with the MDP? It should support other agents
    => agent should only see MDP interface: possible states and actions, initial state, new state and reward after taking an action
    agent should be free to do whatever it wants and using the algorithm just means running the function that is a specific sequence of actions
    agent needs some state to work across episodes

Policy

Action-value function

MDP (models problem)

Episodes (e.g. one round in hide and seek)

Basic example:
per round:
    n-Sarsa interacts with MDP and further learns a policy
    somehow track metrics like policy performance
    get learned policy for evaluations

2-agent example like hide and seek:
per round:
    both interact in rounds after one another with MDP and further learn their policy
    problem: the optimal policy is unknown, but you have a clear win or lose for each agent
    see section hide and seek as an MDP

## hide and seek as an MDP
Two moving players. One hider, one seeker.
Seeker wins if it touches hider while pointing in its direction.
Hider wins after some amount of time.
would this even lead to an interesting game in a plain grid world?
no it would all be about luck. You can't the quality of a hideout in a simple grid world.
This tag running thing would be the closest usable thing, but you'd have to model velocity and maybe inertia.
-> stick to some simple round based game

## More detail
### Agent
depends on Policy -> depends on MDP
```Julia
struct Agent
    pi
end

struct OffPolicyAgent
    b
end


```

### Learning algorithm
Shouldn't depend on specific MDP
```Julia
function nstepQ (agent::Agent, mdp)


end
```

### Policy
depends on MDP
Static out: state space and action space (internally from pointer to its MDP)
todo: is the static out really needed or will the calling side always know what legal states are?
In: state
Out: action
learnable: todo find some good way, e.g. mutable or immutable? learn from action-value function

### Episode
implicitly by specific simulation function

### Simulation
Some custom implementation using all the other parts.

### Action-value-function
Depends on MDP to get state space and action space
Static out: state space and action space
In: state + action
out: value
For a simple grid world example this can be plain matrix.
With a huge state space but only sparse usage something else is needed.

### MDP
Static out: state space, action space
In: state + action
out: state + reward

### State space
Examples: grid world, list of MDP-vertices, hide and seek game
-> different implemetations needed depending on space size, dimension and sparsity
todo

grid world: just coordinates? would be N (where N = number of tiles)





hide and seek:
    coordinates of hider and seeker? no, then they could see each other
    coordinates + coordinate of other agent if it's in line of sight
        this would be N² + N I believe (separate N term is for the additional part-state that the other agent is invisible)
        this needs to depend on time to enable the hider to win

        do I need the entire state space at some point?
        shouldn't it be possible to just store the encountered states, but this could also explode
        do I need the entire action space at some point?


### Action space
Actions are usually very limited, so a list should be fine.