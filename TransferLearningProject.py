"""
Complete Multi-Agent Transfer Learning Experiment
"""
from collections import deque, defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import pickle
from tqdm import tqdm

class EnhancedMultiUAVEnvironment:
    """Optimized environment with shaped rewards"""
    def __init__(self, grid_size=10, num_uavs=2, obstacles=None, mec_nodes=None, goal_positions=None):
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.obstacles = set(obstacles or [(2,2), (5,5)])
        self.mec_nodes = set(mec_nodes or [(3,3), (6,6)])
        self.goal_positions = set(goal_positions or [(9,9)])
        self.uav_positions = [(0, 0)] * num_uavs
        self.max_steps = grid_size * 3
        self.action_map = [(0,1), (0,-1), (-1,0), (1,0)]  # Up, Down, Left, Right

    def reset(self):
        self.uav_positions = [(0, 0)] * self.num_uavs
        self.current_step = 0
        return self._get_state()

    def step(self, actions):
        self.current_step += 1
        individual_rewards = np.zeros(self.num_uavs)
        new_positions = []
        global_reward = 0

        for i in range(self.num_uavs):
            x, y = self.uav_positions[i]
            dx, dy = self.action_map[actions[i]]
            
            new_x = np.clip(x + dx, 0, self.grid_size-1)
            new_y = np.clip(y + dy, 0, self.grid_size-1)

            # Collision and reward calculation
            if (new_x, new_y) in self.obstacles:
                individual_rewards[i] -= 1
                new_positions.append((x, y))
            else:
                dist = self._distance_to_goal((new_x, new_y))
                individual_rewards[i] += 1/(dist + 1e-3)  # Avoid division by zero
                new_positions.append((new_x, new_y))

            # Goal achievement bonus
            if (new_x, new_y) in self.goal_positions:
                individual_rewards[i] += 10
                global_reward += 5

        self.uav_positions = new_positions
        total_reward = np.sum(individual_rewards) + global_reward
        done = self.current_step >= self.max_steps or \
               all(p in self.goal_positions for p in self.uav_positions)

        return self._get_state(), total_reward, done, {}

    def _distance_to_goal(self, position):
        return min(abs(x - position[0]) + abs(y - position[1]) 
                   for (x, y) in self.goal_positions)

    def _get_state(self):
        return tuple(self.uav_positions)

class RobustQLearningAgent:
    """Advanced Q-learning with transfer capabilities"""
    def __init__(self, action_space, transfer_agent=None):
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))
        self.action_space = action_space
        self.alpha = 0.25
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.1
        self.replay_buffer = deque(maxlen=2000)
        
        # Transfer learning initialization
        if transfer_agent:
            self._initialize_transfer(transfer_agent)

    def _initialize_transfer(self, source_agent):
        for state in source_agent.q_table:
            self.q_table[state] = source_agent.q_table[state].copy()
        self.epsilon = max(source_agent.epsilon, 0.2)

    def get_state_key(self, raw_state):
        return tuple((x//2, y//2) for (x, y) in raw_state)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)
        state_key = self.get_state_key(state)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        self.replay_buffer.append( (state, action, reward, next_state) )
        
        if len(self.replay_buffer) >= 64:
            batch = random.sample(self.replay_buffer, 64)
            self._process_batch(batch)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _process_batch(self, batch):
        for s, a, r, ns in batch:
            s_key = self.get_state_key(s)
            ns_key = self.get_state_key(ns)
            
            current = self.q_table[s_key][a]
            max_future = np.max(self.q_table[ns_key])
            new_value = (1 - self.alpha) * current + self.alpha * (r + self.gamma * max_future)
            
            self.q_table[s_key][a] = np.clip(new_value, -50, 50)

def train_agent(env, episodes=1000, transfer_agent=None):
    agent = RobustQLearningAgent(action_space=[0,1,2,3], transfer_agent=transfer_agent)
    reward_history = []
    success_rates = []

    for _ in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        done = False
        success = False

        while not done:
            actions = [agent.choose_action(state) for _ in range(env.num_uavs)]
            next_state, reward, done, _ = env.step(actions)
            
            for action in actions:
                agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            success = any(pos in env.goal_positions for pos in env.uav_positions)

        reward_history.append(total_reward)
        success_rates.append(1 if success else 0)

    return agent, reward_history, success_rates

def run_transfer_comparison():
    # Environment configurations
    source_config = {
        'grid_size': 10,
        'num_uavs': 2,
        'obstacles': [(2,2), (5,5)],
        'goal_positions': [(9,9)]
    }
    
    target_config = {
        'grid_size': 12,
        'num_uavs': 2,
        'obstacles': [(3,3), (4,4), (8,8)],
        'goal_positions': [(11,11)]
    }

    # Train source agent
    print("Training source agent...")
    source_env = EnhancedMultiUAVEnvironment(**source_config)
    source_agent, src_rewards, _ = train_agent(source_env, 1000)

    # Train target with transfer
    print("\nTraining target agent WITH transfer...")
    target_env = EnhancedMultiUAVEnvironment(**target_config)
    transfer_agent, trans_rewards, trans_success = train_agent(target_env, 800, source_agent)

    # Train target without transfer
    print("\nTraining target agent WITHOUT transfer...")
    naive_agent, naive_rewards, naive_success = train_agent(target_env, 800)

    # Plot results
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    plt.plot(trans_rewards, label='With Transfer')
    plt.plot(naive_rewards, label='Without Transfer')
    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.convolve(trans_success, np.ones(50)/50, mode='valid'), label='Transfer Success')
    plt.plot(np.convolve(naive_success, np.ones(50)/50, mode='valid'), label='Naive Success')
    plt.title("Success Rate Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("transfer_comparison.png")
    plt.show()

    # Save results
    results = {
        'source_rewards': src_rewards,
        'transfer_rewards': trans_rewards,
        'naive_rewards': naive_rewards,
        'transfer_success': trans_success,
        'naive_success': naive_success
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/transfer_experiment.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    run_transfer_comparison()
