import cv2
import gym
import numpy as np


def render_rollout(network, env_name, seed, save=False, path=""):
    env = gym.make(env_name)
    env.seed(seed)
    observation = env.reset()
    if save and len(path) > 0:
        import os

        if os.path.isdir(path):
            import shutil

            shutil.rmtree(path)
        os.makedirs(path)

    for t in range(1000):
        # toy-nn doesnt handle having squeezed shapes like (3,) but it handles (3,1) fine
        frame = env.render("rgb_array")
        if save:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path + f"/{t}.jpg", frame)
        observation = np.expand_dims(observation, -1)
        prediction = network.predict(observation)
        action = np.argmax(prediction)
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
