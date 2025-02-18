import numpy as np

states = ["A", "B", "C", "D", "E"]

# Define stupid policy
policy = {
    "A": "a1",
    "B": "b1",
    "C": "c1",
    "D": "d1",
    "E": "e1"
}

# Rewards for each state
rewards = {
    "A": -3,
    "B": -2,
    "C": -2,
    "D": -1,
    "E": -1,
    "F": 10
}

# Transition model for all probabilities of ending in a state with an action
transition_model = {
    "A": {
        "a1": {"A": 0.2, "B": 0.3, "C": 0.5},
        "a2": {"C": 0.2, "D": 0.8}
    },
    "B": {
        "b1": {"A": 0.2, "E": 0.8},
        "b2": {"B": 0.7, "C": 0.3}
    },
    "C": {
        "c1": {"D": 0.6, "F": 0.4},
        "c2": {"B": 0.1, "E": 0.1, "F": 0.8}
    },
    "D": {
        "d1": {"C": 1.0},
        "d2": {"A": 0.6, "F": 0.4}
    },
    "E": {
        "e1": {"A": 0.8, "F": 0.2}
    },
    "F": {
        "terminal": True,
        "reward": 10
    }
}


def policy_evaluation(transition_model, policy, gamma):
    n_states = len(states)

    # Convert all states to indexes for better iteration
    state_to_index = {state: idx for idx, state in enumerate(states)}

    # Setup P probability matrix and R reward vector with zeros
    P = np.zeros((n_states, n_states))
    R = np.zeros(n_states)

    # FIll transitionmatirx ja reward vector
    for state in states:
        i = state_to_index[state]

        # Get action from current policy and the transition probabilities based on that action
        action = policy[state]
        transitions = transition_model[state][action]

        # Iterate through all transitions and fill up (i, j) matrix from the state indices. 
        # This is why states were converted to indices: state is a row (as an index) and the probability is the j
        for next_state, prob in transitions.items():
            if next_state in state_to_index:
                j = state_to_index[next_state]
                P[i][j] = prob
            
        R[i] = rewards[state]

    # print(P) # Check out the cool matrix (i, j)
    
    # Calculate inverse of Pu = r
    I = np.eye(n_states)
    V = np.linalg.solve(I - gamma * P, R)

    # Returns the values for each state as a dictionary
    return dict(zip(states, V))

"""
Complete policy iteration algorithm following the steps:
1. Policy evaluation: compute V for the current policy
2. Policy improvement: update the policy
3. Termination when no changes occur
"""
def policy_iteration(transition_model, policy, V_pi, rewards, gamma):
    V = V_pi.copy() # Copy V so we don't make changes to the original values
    iteration_count = 1 # Keep track of iterations for under the hood examination
    
    # Loop until no change to policy
    while True:
        # Policy evaluation
        V = policy_evaluation(transition_model, policy, gamma)
        policy_stable = True

        # Policy improvement
        for state in states:
            old_action = policy[state]
            action_values = {}

            for action, transitions in transition_model[state].items():
                # Compute Q(s, a), which is the sum of utilities, probabilities and utilities for the next steps 
                q = rewards[state]
                for next_state, prob in transitions.items():
                    if next_state in V:
                        q += gamma * prob * V[next_state]
                    else:
                        q += gamma * prob * rewards['F'] # F is the only terminal state
                # print(state, action,  q) # See the q-values in action!
                action_values[action] = q
                

            # Choose the best action from the Q-values "arg max a"
            best_action = max(action_values, key=action_values.get)
            if best_action != old_action:
                policy_stable = False
                policy[state] = best_action
            
        # If the old policy (or old actions) did not change, we reached optimal policy
        if policy_stable:
            # print(f"Policy converged after {iteration_count} iterations.")
            break
        else:
            iteration_count += 1

    return policy, V

V_pi = policy_evaluation(transition_model, policy, gamma=0.9)
print("Exercise a) answer:")
print(" Original policy:", policy)
print(" Evaluation values:", {s: f"{v:.2f}" for s, v in V_pi.items()})


# Print final results
print("Exercise b) answer:")
optimal_policy, optimal_values = policy_iteration(transition_model, policy, V_pi, rewards, gamma=0.9)
print(" Optimal policy:", optimal_policy)
print(" Optimal values:", {s: f"{v:.2f}" for s, v in optimal_values.items()})