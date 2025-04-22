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
        # 保存源知识以供参考
        self.source_knowledge = {}
        for state in source_agent.q_table:
            self.q_table[state] = source_agent.q_table[state].copy()
            self.source_knowledge[state] = source_agent.q_table[state].copy()
        
        self.epsilon = max(source_agent.epsilon, 0.2)

    def get_state_key(self, raw_state):
        return tuple((x//2, y//2) for (x, y) in raw_state)

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        
        # 检查是否为"新"状态（源智能体未见过或很少见过）
        is_novel_state = state_key not in self.q_table or np.max(np.abs(self.q_table[state_key])) < 0.1
        
        # 针对新状态提高探索率
        exploration_rate = self.epsilon * 1.5 if is_novel_state else self.epsilon
        
        if np.random.rand() < exploration_rate:
            return random.choice(self.action_space)
        
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        
        # 早期使用更高的学习率
        if len(self.replay_buffer) < 500:
            original_alpha = self.alpha
            self.alpha = min(0.5, self.alpha * 1.5)  # 加速早期学习
            
        if len(self.replay_buffer) >= 64:
            batch = random.sample(self.replay_buffer, 64)
            self._process_batch(batch)
            
        # 恢复正常学习率
        if len(self.replay_buffer) < 500:
            self.alpha = original_alpha
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _process_batch(self, batch):
        for s, a, r, ns in batch:
            s_key = self.get_state_key(s)
            ns_key = self.get_state_key(ns)

            current = self.q_table[s_key][a]
            max_future = np.max(self.q_table[ns_key])
            
            # 标准Q学习更新
            q_update = (1 - self.alpha) * current + self.alpha * (r + self.gamma * max_future)
            
            # 如果有源智能体知识，添加知识蒸馏项
            if hasattr(self, 'source_knowledge') and s_key in self.source_knowledge:
                source_value = self.source_knowledge[s_key][a]
                distill_weight = 0.3  # 早期较高，后期降低
                new_value = (1-distill_weight) * q_update + distill_weight * source_value
            else:
                new_value = q_update

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
    source_agent, src_rewards, _ = train_agent(source_env, 10000)

    # Train target with transfer
    print("\nTraining target agent WITH transfer...")
    target_env = EnhancedMultiUAVEnvironment(**target_config)
    transfer_agent, trans_rewards, trans_success = train_agent(target_env, 600, source_agent)

    # Train target without transfer
    print("\nTraining target agent WITHOUT transfer...")
    naive_agent, naive_rewards, naive_success = train_agent(target_env, 600)

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

def perform_statistical_analysis():
    from scipy import stats
    import numpy as np
    
    with open("results/transfer_experiment.pkl", "rb") as f:
        data = pickle.load(f)

    trans_rewards = np.array(data['transfer_rewards'][-100:])
    naive_rewards = np.array(data['naive_rewards'][-100:])
    trans_success = np.array(data['transfer_success'][-100:])
    naive_success = np.array(data['naive_success'][-100:])


    t_reward, p_reward = stats.ttest_ind(trans_rewards, naive_rewards)
    reward_improvement = (np.mean(trans_rewards) - np.mean(naive_rewards)) / np.mean(naive_rewards) * 100

    u_stat, p_success = stats.mannwhitneyu(trans_success, naive_success)
    success_rate_diff = np.mean(trans_success) - np.mean(naive_success)

    pooled_std = np.sqrt((np.std(trans_rewards)**2 + np.std(naive_rewards)**2))/2
    cohen_d = (np.mean(trans_rewards) - np.mean(naive_rewards)) / pooled_std

    print("\n===== Statistical Analysis Report =====")
    print(f"1. Reward Comparison (t-test):")
    print(f"   t-statistic = {t_reward:.3f}, p-value = {p_reward:.4f}")
    print(f"   Mean improvement: {reward_improvement:.1f}%")
    
    print(f"\n2. Success Rate Comparison (Mann-Whitney U):")
    print(f"   U-statistic = {u_stat:.0f}, p-value = {p_success:.4f}")
    print(f"   Absolute difference: {success_rate_diff:.3f}")
    
    print(f"\n3. Effect Size:")
    print(f"   Cohen's d = {cohen_d:.3f}")
    print("="*40)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([naive_rewards, trans_rewards], 
                labels=['Baseline', 'Transfer Learning'])
    plt.title('Reward Distribution Comparison')
    plt.ylabel('Cumulative Reward')
    
    plt.subplot(1, 2, 2)
    success_rates = [np.mean(naive_success), np.mean(trans_success)]
    plt.bar(['Baseline', 'Transfer'], success_rates)
    plt.title('Success Rate Comparison')
    plt.ylabel('Success Probability')
    
    plt.tight_layout()
    plt.savefig("statistical_analysis.png")
    plt.show()

def analyze_early_convergence():
    """Analyze early success rates and convergence speed between transfer and naive learning"""
    from scipy import stats
    import numpy as np
    
    with open("results/transfer_experiment.pkl", "rb") as f:
        data = pickle.load(f)

    trans_rewards = np.array(data['transfer_rewards'])
    naive_rewards = np.array(data['naive_rewards'])
    trans_success = np.array(data['transfer_success'])
    naive_success = np.array(data['naive_success'])
    
    # Get actual episode counts
    num_episodes = min(len(trans_rewards), len(naive_rewards))
    early_window = min(num_episodes, 100)  # Shortened to first 200 episodes
    
    # Calculate early-stage success rate
    early_trans_success = np.mean(trans_success[:early_window])
    early_naive_success = np.mean(naive_success[:early_window])
    
    # Use sliding window to determine convergence point (standard deviation < 0.5 in 50-episode window)
    window_size = 50
    convergence_threshold = 0.5
    
    def find_convergence_point(rewards):
        for i in range(len(rewards) - window_size):
            window = rewards[i:i+window_size]
            if np.std(window) < convergence_threshold:
                return i
        return len(rewards)  # If no convergence, return total episodes
    
    trans_convergence = find_convergence_point(trans_rewards)
    naive_convergence = find_convergence_point(naive_rewards)
    
    print("\n===== Early Learning & Convergence Analysis =====")
    print(f"1. Early Success Rate Comparison (First {early_window} episodes):")
    print(f"   Transfer Learning Success Rate: {early_trans_success:.3f}")
    print(f"   Baseline Learning Success Rate: {early_naive_success:.3f}")
    print(f"   Improvement: {(early_trans_success - early_naive_success):.3f} ({(early_trans_success/early_naive_success - 1)*100:.1f}%)")
    
    print(f"\n2. Convergence Speed Comparison:")
    print(f"   Transfer Learning Convergence Point: Episode {trans_convergence}")
    print(f"   Baseline Learning Convergence Point: Episode {naive_convergence}")
    if trans_convergence >= num_episodes and naive_convergence >= num_episodes:
        print(f"   Note: Neither method converged within the {num_episodes} episode limit")
    else:
        print(f"   Earlier Convergence by: {naive_convergence - trans_convergence} episodes")
    print("="*40)

    # Visualize results
    plt.figure(figsize=(15, 6))
    
    # Early success rate comparison
    plt.subplot(1, 2, 1)
    window = 20  # Smaller window for early episodes visualization
    
    # Calculate convolution results first
    trans_conv = np.convolve(trans_success, np.ones(window)/window, mode='valid')
    naive_conv = np.convolve(naive_success, np.ones(window)/window, mode='valid')
    
    # Create matching x-ranges for plotting
    x_trans = np.arange(window, window + len(trans_conv))[:len(trans_conv)]
    x_naive = np.arange(window, window + len(naive_conv))[:len(naive_conv)]
    
    # Plot with matching dimensions
    plt.plot(x_trans, trans_conv, label='Transfer Learning', color='blue')
    plt.plot(x_naive, naive_conv, label='Baseline', color='red')
    
    # Mark the early window boundary
    if early_window >= window:
        plt.axvline(x=early_window, linestyle='--', color='gray', alpha=0.5)
    
    # Set x-axis limit to focus on early episodes
    plt.xlim(window, early_window + 50)  # Show a bit beyond the early window
    
    plt.title('Early Success Rate Comparison (Moving Average)')
    plt.xlabel('Training Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    
    # Convergence point in reward curves
    plt.subplot(1, 2, 2)
    smooth_window = 20  # Smaller window for clearer early episode visualization
    
    # Calculate reward convolutions 
    trans_reward_conv = np.convolve(trans_rewards, np.ones(smooth_window)/smooth_window, mode='valid')
    naive_reward_conv = np.convolve(naive_rewards, np.ones(smooth_window)/smooth_window, mode='valid')
    
    # Create matching x-ranges for rewards
    x_trans_r = np.arange(smooth_window, smooth_window + len(trans_reward_conv))
    x_naive_r = np.arange(smooth_window, smooth_window + len(naive_reward_conv))
    
    plt.plot(x_trans_r, trans_reward_conv, label='Transfer Learning', color='blue')
    plt.plot(x_naive_r, naive_reward_conv, label='Baseline', color='red')
    
        # Mark convergence points
    if trans_convergence >= smooth_window:
        plt.axvline(x=trans_convergence, linestyle='--', color='blue', alpha=0.7)
    if naive_convergence <= early_window + 50 and naive_convergence >= smooth_window:
        plt.axvline(x=naive_convergence, linestyle='--', color='red', alpha=0.7)
    
    plt.title('Early Learning Reward Comparison (Moving Average)')
    plt.xlabel('Training Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("early_convergence_analysis.png")
    plt.show()

if __name__ == "__main__":
    run_transfer_comparison()
    perform_statistical_analysis()
    analyze_early_convergence()



if __name__ == "__main__":
    run_transfer_comparison()
