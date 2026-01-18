import gymnasium
import flappy_bird_gymnasium
import numpy as np
import time
import utils
import numpy as np
import time # Added time import as it's used later
import pickle

# Load Q-Table
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print(f"Loaded Q-Table with {len(q_table)} states.")
except FileNotFoundError:
    print("Error: q_table.pkl not found. Train the agent first.")
    exit()

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

state, _ = env.reset()

# Play Loop
while True:
    state_key = utils.get_discrete_state(state)
    
    # Check if state exists in Q-table
    if state_key in q_table:
        action = np.argmax(q_table[state_key])
    else:
        action = 0
        
    state, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.03) # Added time.sleep back in

    if terminated or truncated:
        state, _ = env.reset()

env.close()