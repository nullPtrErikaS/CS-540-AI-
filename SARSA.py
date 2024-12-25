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

        if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax([Q_table[(state, a)] for a in range(env.action_space.n)])

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with SARSA

        while (not done):
            # Take action and observe the result
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Choose next action using epsilon-greedy
            if random.uniform(0, 1) < EPSILON:
                next_action = env.action_space.sample()  # Explore
            else:
                next_action = np.argmax([Q_table[(next_state, a)] for a in range(env.action_space.n)])

            # Update Q-value using SARSA update rule
            old_value = Q_table[(state, action)]
            next_value = Q_table[(next_state, next_action)]
            new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_value - old_value)
            Q_table[(state, action)] = new_value

            # Update for next step
            state = next_state
            action = next_action
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
    model_file = open(f'Q_TABLE_SARSA.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################