# import gymnasium as gym
# import time
# import numpy as np

# # Create the environment with rendering enabled
# env = gym.make("CarRacing-v2", render_mode="human")

# # --- Inspect spaces ---
# print("=== CarRacing-v2 Info ===")
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)
# print("\nExample observation shape:", env.observation_space.shape)
# print("Example action:", env.action_space.sample())
# print("==========================\n")

# # Reset environment
# obs, info = env.reset(seed=42)
# total_reward = 0

# # --- Run a few steps ---
# for step in range(1000000):
#     # Sample a random action (you can later replace this with your policy)
#     action = env.action_space.sample()

#     # Step through the environment
#     obs, reward, terminated, truncated, info = env.step(action)
#     total_reward += reward

#     # Display details
#     print(f"Step {step + 1}")
#     print("  Action:", np.round(action, 3))  # throttle, steering, brake
#     print("  Observation shape:", obs.shape)
#     print("  Reward:", round(reward, 3))
#     print("  Done:", terminated or truncated)
#     print("---------------------------")

#     # Render
#     env.render()
#     time.sleep(0.03)

#     # if terminated or truncated:
#     #     print("Episode finished — resetting environment.\n")
#     #     obs, info = env.reset()
#     #     total_reward = 0

# env.close()


import gymnasium as gym

# Use v3 instead of v2
env = gym.make("CarRacing-v2", render_mode="human")  
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample() 
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs.shape)
    print(reward)
    print("******************")    # done = terminated or truncated  # uncomment this line to end episodes properly

env.close()
