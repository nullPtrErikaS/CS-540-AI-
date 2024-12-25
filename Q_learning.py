# Model: Gpt 4o
#: Prompts: 
# how should i go about doing this?
# how would you do this?
# why am i gettign this syntax error 
# what is the best way to do this?
# what is wrong with my code?

import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        state = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  # Explore: choose random action
            else:
                # Exploit: choose the action with the highest Q-value
                action = np.argmax([Q_table[(state, a)] for a in range(env.action_space.n)])

            # Take the action and observe the result
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-Learning Update
            old_value = Q_table[(state, action)]
            next_max = np.max([Q_table[(next_state, a)] for a in range(env.action_space.n)])
            new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
            Q_table[(state, action)] = new_value

            state = next_state
            episode_reward += reward

        # Decay epsilon after each episode
        EPSILON *= EPSILON_DECAY
            
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################