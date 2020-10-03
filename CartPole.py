import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CartPoleQAgent():
    def __init__(self, buckets=(3, 3, 6, 6),
                 num_episodes=500, min_lr=0.1,
                 min_epsilon=0.1, discount=1.0, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make('CartPole-v0')

        # This is the action-value function being initialized to 0's
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

        # [position, velocity, angle, angular velocity]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2],
                             math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2],
                             -math.radians(50) / 1.]

        #
        self.steps = np.zeros(self.num_episodes)

    def discretize_state(self, obs):
       #Code for making discrete values out of the continuous space
        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(self.lower_bounds[i]))
                       / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def choose_action(self, state):
        
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def get_action(self, state, e):
        #Gives the probability of each action.
        obs = self.discretize_state(state)
        action_vector = self.Q_table[obs]
        epsilon = self.get_epsilon(e)
        action_vector = self.normalize(action_vector, epsilon)
        return action_vector

    def normalize(self, action_vector, epsilon):
        

        total = sum(action_vector)
        new_vector = (1 - epsilon) * action_vector / (total)
        new_vector += epsilon / 2.0
        return new_vector

    def update_q(self, state, action, reward, new_state):
        """
        Updates Q-table using the rule as described by Sutton and Barto in
        Reinforcement Learning.
        """
        self.Q_table[state][action] += (self.learning_rate *
                                        (reward
                                         + self.discount * np.max(self.Q_table[new_state])
                                         - self.Q_table[state][action]))

    def get_epsilon(self, t):
        """Gets value for epsilon. It declines as we advance in episodes."""
        # Ensures that there's almost at least a min_epsilon chance of randomly exploring
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        """Gets value for learning rate. It declines as we advance in episodes."""
        # Learning rate also declines as we add more episodes
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        #This function is responsible for training of the agent
        # Looping for each episode
        for e in range(self.num_episodes):
            # Initializes the state
            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            # Looping for each step
            while not done:
                self.steps[e] += 1
                # Choose A from S
                action = self.choose_action(current_state)
                # Take action
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                # Update Q(S,A)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state

                # We break out of the loop when done is False which is
                # a terminal state.
        print('Finished training!')

    def plot_learning(self):
        """
        Plots the number of steps at each episode and prints the
        amount of times that an episode was successfully completed.
        """
        sns.lineplot(range(len(self.steps)), self.steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.show()
        t = 0
        for i in range(self.num_episodes):
            if self.steps[i] == 200:
                t += 1
        print(t, "episodes were successfully completed.")

    def run(self):
        """Runs an episode while displaying the cartpole environment."""
        self.env = gym.wrappers.Monitor(self.env, 'cartpole')
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
            self.env.render()
            t = t + 1
            action = self.choose_action(current_state)
            obs, reward, done, _ = self.env.step(action)
            new_state = self.discretize_state(obs)
            current_state = new_state

        return t

def load_q_learning():
    agent = CartPoleQAgent()
    agent.train()
    agent.plot_learning()

    return agent

agent = load_q_learning()

#code for Random Policy
def linear(obs, vec):
    control = np.sum(obs*vec)
    action = 1 if control>0 else 0
    return(action)

env = gym.make('CartPole-v0') #making the environment

best_vec, best_score = np.zeros(4), 0 #making the Q-table
num_draws = 50
for k in range(num_draws):
    vec = np.random.uniform(low=-1, high=1, size=4) #Taking Random values
    avg_reward = 0
    num_eval_eps = 1
    for i in range(num_eval_eps): #Steps in each episode
        ep_reward = 0
        obs = env.reset()
        done = False
        while not done:
            action = linear(obs, vec)
            obs, reward, done, info = env.step(action) #Taking the action
            ep_reward += reward #adding the reward
            if done:
                avg_reward += ep_reward/num_eval_eps
                ep_reward = 0
    if avg_reward > best_score:
        best_score, best_vec = avg_reward, vec

env.close()

print('Best score {}\nBest vec {}'.format(best_score, best_vec))

#Q Learning Code Reference: https://github.com/JoeSnow7/Reinforcement-Learning/blob/master/Cartpole%20Q-learning.ipynb
#Random Code Reference : https://github.com/Ankur-Deka/RL-Experiments/blob/master/Cartpole_random_search.ipynb
