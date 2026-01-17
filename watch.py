import gymnasium
import flappy_bird_gymnasium
import numpy as np
import time
import utils


try:
    q_table = np.load("brain.npy")
    print("Q-Table created")
except:
    print("First Run train.py")
    exit()


env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

state, _ = env.reset()
state_disc = utils.get_discrete_state(state)

while True:
    
    action = np.argmax(q_table[state_disc])

    state, reward, terminated, truncated, info = env.step(action)
    state_disc = utils.get_discrete_state(state)
    
    time.sleep(0.03)

    if terminated:
        state, _ = env.reset()
        state_disc = utils.get_discrete_state(state)