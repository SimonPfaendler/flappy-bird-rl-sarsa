# Flappy Bird RL Agent

A Reinforcement Learning project using **Q-Learning with Eligibility Traces** to play Flappy Bird.

## Algorithm: Q(λ) with Replacing Traces
Tabular Q-Learning approach enhanced with:
- **Eligibility Traces (λ=0.7)**: Accelerates learning by updating past states responsible for current rewards (using "replacing traces").
- **Optimistic Initialization**: Q-values start at 0.1 to encourage early exploration.
- **Dynamic Alpha**: Learning rate decays when high scores are achieved to stabilize convergence.


### Frameskipping
To stabilize flight paths, I treat "Jump" as a macro-action.
- **Micro-action**: Agent jumps once.
- **Macro-behavior**: Agent jumps and then idles (skips inputs) for **3 frames**.
- This reduces "jittery" flapping and allows the agent to see the result of its jump action over a longer time horizon.

### State Discretization
Using the *FlapAI Bird* methodology with custom offsets:
- **Vertical Distance (`dist_y`)**: Calculated relative to a specific offset from the bottom pipe to encourage safe passage.
  - Formula: `dist_y = state[9] - (state[5] + 0.01)`
- **Discretization Bins**:
  - `dist_x`: 0.1 precision
  - `dist_y`: 0.1 precision
  - `velocity`: 1.0 precision

## Hyperparameters
- **Episodes**: 10,000
- **Alpha (Learning Rate)**: 0.07
- **Gamma (Discount)**: 0.99
- **Epsilon**: 0.0 (Greedy, relies on optimistic init)
- **Jump Skip Frames**: 3

## File Structure
- `train.py`: Main training loop, Q-Learning update rule, and frameskipping logic.
- `utils.py`: Helper functions for state discretization.
- `analyse.py`: Tools to view and analyze training logs.
- `watch.py`: Script to watch the trained agent play.