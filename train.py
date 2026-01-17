import gymnasium
import flappy_bird_gymnasium
import numpy as np
import random


env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)


horiz_bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.5])
vert_bins = np.linspace(-0.5, 1.5, 14) 
vel_bins = np.array([-2.0, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.2, 0.4, 0.6, 1.0])

state_dims = (len(horiz_bins)+1, len(vert_bins)+1, len(vel_bins)+1)
q_table = np.zeros(state_dims + (env.action_space.n,))

def get_discrete_state(state):
    horiz_dist = state[3]
    vert_dist = state[5] - state[9] 
    velocity = state[10]
    
    x = np.digitize(horiz_dist, horiz_bins)
    y = np.digitize(vert_dist, vert_bins)
    v = np.digitize(velocity, vel_bins)
    return (x, y, v)


episodes = 200000       
epsilon = 1.0          
epsilon_min = 0.01
epsilon_decay = 0.9997 
alpha = 0.1    
gamma = 0.99   
lam = 0.3

print("Training")

for episode in range(episodes):
    state, _ = env.reset()
    state_disc = get_discrete_state(state)
    
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state_disc])

    terminated = False
    step_count = 0
    
    # Eligibility Trace
    eligibility_trace = np.zeros_like(q_table)

    while not terminated:
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_disc = get_discrete_state(next_state)

        # --- REWARD SYSTEM ---
        vert_diff = next_state[5] - next_state[9]

        if terminated:
            reward = -100 
        elif reward > 0.9: 
            reward = 200  
        else:
            reward = 0.1

        #Epsilon-Greedy
        if random.random() < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state_disc])

        # SARSA(Lambda) Update
        
        # 1. TD-Fehler
        current_q = q_table[state_disc + (action,)]
        if terminated:
             target = reward
        else:
             next_q_val = q_table[next_state_disc + (next_action,)]
             target = reward + gamma * next_q_val
        
        delta = target - current_q

        # 2. Eligibility Trace increment
        # Accumulating Traces
        eligibility_trace[state_disc + (action,)] += 1

        # 3. Q-Table Update
        q_table += alpha * delta * eligibility_trace

        # 4. Eligibility Trace Decay
        eligibility_trace *= gamma * lam

        state_disc = next_state_disc
        action = next_action
        step_count += 1
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 2000 == 0:
        print(f"Episode: {episode}, Score: {step_count}, Epsilon: {epsilon:.4f}")


np.save("brain.npy", q_table)
print("Training done")
env.close()