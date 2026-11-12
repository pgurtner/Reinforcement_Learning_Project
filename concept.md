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
 * conceptualize code
 * roadmap
 * timeplan including report (5 days including today/thursday and excluding tuesday)

# code concept
 * Algorithm
 * MDP
 * example / problem
 * rest of the code is independent of the individual problem

Actor:
    plays according to learning algorithm and afterwards according to learned policy
    each action interacts with the environment -> or more generally with the MDP? It should support other actors
    => actor should only see MDP interface: possible states and actions, initial state, new state and reward after taking an action
    actor should be free to do whatever it wants and using the algorithm just means running the function that is a specific sequence of actions
    actor needs some state to work across episodes

Policy

Action-value function

MDP (models problem)

Episodes (e.g. one round in hide and seek)

Basic example:
per round:
    n-Sarsa interacts with MDP and further learns a policy
    somehow track metrics like policy performance
    get learned policy for evaluations

2-actor example like hide and seek:
per round:
    both interact in rounds after one another with MDP and further learn their policy
    problem: the optimal policy is unknown, but you have a clear win or lose for each actor
    todo: how to model hide and seek as an MDP

## More detail
### Actor
depends on Policy -> depends on MDP
```Julia
struct Actor
    pi
end

struct OffPolicyActor
    b
end


```

### Learning algorithm
Shouldn't depend on specific MDP
```Julia
function nstepQ (actor::OnPolicyActor, mdp)


end
```

### Policy
depends on MDP -> todo: should it store a pointer to its MDP?
Static out: MDP to get state space and action space
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

### Action space
Actions are usually very limited, so a list should be fine.