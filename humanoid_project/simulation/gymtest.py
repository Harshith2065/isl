"""Simple Gymnasium environment test script."""

import gymnasium as gym  # type: ignore

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for _ in range(200):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    env.render()
    if done or truncated:
        env.reset()

env.close()
