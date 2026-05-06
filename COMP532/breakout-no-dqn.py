import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import ale_py

gym.register_envs(ale_py)

#DQN using cnn
class QNetwork(nn.Module):
    def __init__(self, action_size):
        

#dqn agent
class DQNAgent:
    #initialise parameters:
    #gamma, epsilon, lr, etc...

    #funtion to store the agent's experience

    #exploration/exploitation

    #experience replay


#train the agent
def train_dqn():
    env = gym.make("ALE/Breakout-v5", render_mode="human") #render mode human makes the training process actually entertaining
    action_size = env.action_space.n #number of possible actions
    agent = DQNAgent(action_size)
    episodes = 500

    #trianing loop
    for e in range(episodes):
        state, _ = env.reset() #reset the environment to start a new episode
        state = preprocess(state) #call the preprocess function to process the image for the dqn
        state = np.stack([state] * 4, axis=0) #stack the preprocessed state to simulate a 4-frame input (this is required for atari envs)

        done = False
        total_reward = 0
        
        while not done:
            env.render()
            action = agent.act(state) #select an action (exploration/exploitation)

            next_state, reward, terminated, truncated, _ = env.step(action) #take action in env, observe next state and reward
            next_state = preprocess(next_state) #preprocess next state
            next_state = np.append(state[1:], np.expand_dims(next_state, 0), axis=0) #stack

            done = terminated or truncated

            #store experience
            agent.remember(state, action, reward, next_state, done)

            #update the state, add reward to the current total of this episode
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
        
        agent.replay()
    
    env.close()

#process the raw image using cv2
#convert to grayscale -> resize -> normalise
#used for DQN
def preprocess(image):
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 84))
    return image / 255.0

if __name__ == "__main__":
    train_dqn()
