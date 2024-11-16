from SingleDotProblemStaticMonster import Agent, State, Problem
from problemGraphicsMonster import pacmanGraphic # type: ignore
import random

# Initialize Problem
p = Problem('singleDotSmallMonster.txt')  # Change to 'singleDotMedium.txt' for larger maze

# Value Iteration function
def value_iteration(p, gamma=0.9, theta=0.01):
    # Initialize value function (V) with 0 for all states
    V = {}
    for y in range(p.yMax):
        for x in range(p.xMax):
            V[State((x, y))] = 0
    
    # Initialize a policy dictionary
    policy = {}
    
    while True:
        delta = 0
        # Iterate through all the states
        for state in V:
            if p.isTerminal(state):
                continue  # Skip terminal states

            # Value Iteration update rule
            old_v = V[state]
            new_v = float('-inf')
            best_action = None

            # Check all possible moves
            for next_state, action in p.transition(state):
                # Calculate Q(s, a) = R(s) + γ * V(s')
                reward = p.reward(next_state)
                v = reward + gamma * V[next_state]
                if v > new_v:
                    new_v = v
                    best_action = action
            
            V[state] = new_v
            policy[state] = best_action
            delta = max(delta, abs(old_v - new_v))
        
        # Check for convergence
        if delta < theta:
            break
    
    return policy, V

# Apply Value Iteration to find the optimal policy
policy, V = value_iteration(p)

# Visualize the policy and agent's movements
pac = pacmanGraphic(1300, 700)
pac.setup(p)

# Draw the policy (arrows)
for state in policy:
    s = None
    if policy[state] == 'L': s = '\u2190'  # Left arrow
    if policy[state] == 'R': s = '\u2192'  # Right arrow
    if policy[state] == 'U': s = '\u2191'  # Up arrow
    if policy[state] == 'D': s = '\u2193'  # Down arrow
    if s:
        pac.addText(state.agentClass.pos[0] + 0.5, state.agentClass.pos[1] + 0.5, s, fontSize=20)

# Apply the optimal policy to guide Pacman through the maze
print('Apply policy')
currentState = p.getStartState()
count = 0
while currentState:
    a = policy.get(currentState, None)
    if a is None: break  # No valid move, break the loop

    count += 1
    dx, dy = p.potential_moves[a]
    agentPos = (currentState.agentClass.pos[0] + dx, currentState.agentClass.pos[1] + dy)

    if agentPos in p.dots:
        index = p.dots.index(agentPos)
        pac.remove_dot(index)  # Remove dot after collection
    
    currentState = State(agentPos)
    pac.move_pacman(dx, dy)  # Move Pacman on the graphic display
    print(f'Plan length = {count}')
    
    # Check if the agent has reached a terminal state (dot or monster)
    if p.isTerminal(currentState):
        break

