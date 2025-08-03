"""
Reinforcement Learning Agent for ISA Optimization
Uses RL to autonomously explore and evolve custom instruction combinations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Any
import json

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ISAEnvironment:
    """Custom environment for ISA optimization using RL"""
    
    def __init__(self, target_workload: Dict[str, Any]):
        self.target_workload = target_workload
        self.instruction_space = self._create_instruction_space()
        self.current_isa_config = []
        self.baseline_metrics = self._calculate_baseline_metrics()
        self.max_instructions = 10
        self.step_count = 0
        self.max_steps = 50
        
    def _create_instruction_space(self):
        """Define the space of possible custom instructions"""
        return {
            'conv_ops': ['VCONV.8', 'VCONV.16', 'VCONV.F', 'VCONV.SPARSE'],
            'matmul_ops': ['VMMUL.8', 'VMMUL.16', 'VMMUL.F', 'VMMUL.BLOCK'],
            'activation_ops': ['RELU.V', 'RELU.6', 'RELU.L', 'SIGMOID.V', 'TANH.V'],
            'normalization_ops': ['BNORM.2D', 'BNORM.1D', 'LNORM.V'],
            'pooling_ops': ['VPOOL.MAX', 'VPOOL.AVG', 'VPOOL.GAP'],
            'memory_ops': ['VMLOAD', 'VMSTORE', 'PREFETCH.NN'],
            'quantization_ops': ['QUANT.8', 'QUANT.4', 'DEQUANT.V']
        }
    
    def _calculate_baseline_metrics(self):
        """Calculate baseline performance metrics"""
        layers = self.target_workload.get('layers', [])
        total_time = sum(layer.get('avg_time', 0) for layer in layers)
        total_flops = sum(layer.get('flops_estimate', 0) for layer in layers)
        total_params = sum(layer.get('parameters', 0) for layer in layers)
        
        return {
            'execution_time': total_time,
            'flops': total_flops,
            'parameters': total_params,
            'energy': total_flops * 1e-9,  # Estimated energy
            'memory_bandwidth': total_params * 4 * 2  # Estimated memory traffic
        }
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_isa_config = []
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        # Create state vector representing current ISA configuration and workload
        state = np.zeros(64)  # Fixed size state vector
        
        # Workload characteristics (first 20 elements)
        layers = self.target_workload.get('layers', [])
        for i, layer in enumerate(layers[:10]):  # Top 10 layers
            if i < 10:
                state[i*2] = layer.get('percentage', 0) / 100.0
                state[i*2 + 1] = min(layer.get('flops_estimate', 0) / 1e6, 1.0)
        
        # Current ISA configuration (next 24 elements)
        all_instructions = []
        for category in self.instruction_space.values():
            all_instructions.extend(category)
        
        for i, instruction in enumerate(self.current_isa_config):
            if i < 10 and instruction in all_instructions:
                idx = all_instructions.index(instruction)
                if 20 + idx < 64:
                    state[20 + idx] = 1.0
        
        # Performance metrics (last 20 elements)
        if len(self.current_isa_config) > 0:
            current_metrics = self._calculate_current_metrics()
            state[44] = min(current_metrics['speedup'], 5.0) / 5.0
            state[45] = min(current_metrics['energy_reduction'], 1.0)
            state[46] = min(current_metrics['flops_reduction'], 1.0)
            state[47] = len(self.current_isa_config) / self.max_instructions
        
        return state
    
    def step(self, action):
        """Take action in environment"""
        self.step_count += 1
        
        # Decode action
        instruction = self._decode_action(action)
        
        # Add instruction if valid and not duplicate
        if instruction and instruction not in self.current_isa_config:
            if len(self.current_isa_config) < self.max_instructions:
                self.current_isa_config.append(instruction)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps or 
                len(self.current_isa_config) >= self.max_instructions)
        
        next_state = self._get_state()
        
        return next_state, reward, done, {}
    
    def _decode_action(self, action):
        """Decode numeric action to instruction name"""
        all_instructions = []
        for category in self.instruction_space.values():
            all_instructions.extend(category)
        
        if 0 <= action < len(all_instructions):
            return all_instructions[action]
        return None
    
    def _calculate_current_metrics(self):
        """Calculate current performance metrics with ISA extensions"""
        if not self.current_isa_config:
            return {'speedup': 1.0, 'energy_reduction': 0.0, 'flops_reduction': 0.0}
        
        # Simulate performance improvements based on instructions
        total_speedup = 1.0
        energy_reduction = 0.0
        flops_reduction = 0.0
        
        for instruction in self.current_isa_config:
            # Get instruction benefits based on workload
            benefits = self._get_instruction_benefits(instruction)
            total_speedup *= benefits['speedup']
            energy_reduction += benefits['energy_reduction']
            flops_reduction += benefits['flops_reduction']
        
        return {
            'speedup': total_speedup,
            'energy_reduction': min(energy_reduction, 0.8),  # Cap at 80%
            'flops_reduction': min(flops_reduction, 0.7)     # Cap at 70%
        }
    
    def _get_instruction_benefits(self, instruction):
        """Get benefits of specific instruction for current workload"""
        layers = self.target_workload.get('layers', [])
        
        # Default benefits
        benefits = {'speedup': 1.0, 'energy_reduction': 0.0, 'flops_reduction': 0.0}
        
        for layer in layers:
            layer_type = layer.get('type', '').lower()
            layer_weight = layer.get('percentage', 0) / 100.0
            
            # Match instruction to layer type
            if 'conv' in instruction.lower() and 'conv' in layer_type:
                benefits['speedup'] += 0.3 * layer_weight
                benefits['energy_reduction'] += 0.2 * layer_weight
                benefits['flops_reduction'] += 0.15 * layer_weight
            elif 'mmul' in instruction.lower() and ('linear' in layer_type or 'gemm' in layer_type):
                benefits['speedup'] += 0.25 * layer_weight
                benefits['energy_reduction'] += 0.15 * layer_weight
                benefits['flops_reduction'] += 0.12 * layer_weight
            elif 'relu' in instruction.lower() and 'relu' in layer_type:
                benefits['speedup'] += 0.15 * layer_weight
                benefits['energy_reduction'] += 0.1 * layer_weight
                benefits['flops_reduction'] += 0.08 * layer_weight
        
        return benefits
    
    def _calculate_reward(self):
        """Calculate reward for current ISA configuration"""
        if not self.current_isa_config:
            return 0.0
        
        metrics = self._calculate_current_metrics()
        
        # Multi-objective reward function
        speedup_reward = (metrics['speedup'] - 1.0) * 10.0
        energy_reward = metrics['energy_reduction'] * 5.0
        flops_reward = metrics['flops_reduction'] * 3.0
        
        # Penalty for too many instructions (complexity)
        complexity_penalty = len(self.current_isa_config) * 0.1
        
        # Bonus for instruction diversity
        categories_used = set()
        for instruction in self.current_isa_config:
            for category, instructions in self.instruction_space.items():
                if instruction in instructions:
                    categories_used.add(category)
        diversity_bonus = len(categories_used) * 0.5
        
        total_reward = (speedup_reward + energy_reward + flops_reward + 
                       diversity_bonus - complexity_penalty)
        
        return total_reward
    
    def get_action_space_size(self):
        """Get size of action space"""
        total_instructions = sum(len(instructions) for instructions in self.instruction_space.values())
        return total_instructions

class DQNAgent:
    """Deep Q-Network agent for ISA optimization"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
    
    def _build_model(self):
        """Build DQN model"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class ISAOptimizer:
    """Main ISA optimization controller using RL"""
    
    def __init__(self):
        self.env = None
        self.agent = None
        self.training_history = []
    
    def optimize_isa(self, workload_data: Dict[str, Any], episodes: int = 100):
        """Optimize ISA for given workload using RL"""
        self.env = ISAEnvironment(workload_data)
        state_size = 64
        action_size = self.env.get_action_space_size()
        self.agent = DQNAgent(state_size, action_size)
        
        best_reward = float('-inf')
        best_config = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train agent
            if len(self.agent.memory) > 32:
                self.agent.replay()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            # Track best configuration
            if total_reward > best_reward:
                best_reward = total_reward
                best_config = self.env.current_isa_config.copy()
            
            self.training_history.append({
                'episode': episode,
                'reward': total_reward,
                'epsilon': self.agent.epsilon,
                'config_length': len(self.env.current_isa_config)
            })
            
            if episode % 20 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, "
                      f"Best: {best_reward:.2f}, Config: {len(best_config)} instructions")
        
        return {
            'best_config': best_config,
            'best_reward': best_reward,
            'training_history': self.training_history,
            'final_metrics': self.env._calculate_current_metrics() if best_config else {}
        }
    
    def evaluate_config(self, workload_data: Dict[str, Any], isa_config: List[str]):
        """Evaluate a specific ISA configuration"""
        env = ISAEnvironment(workload_data)
        env.current_isa_config = isa_config
        metrics = env._calculate_current_metrics()
        reward = env._calculate_reward()
        
        return {
            'metrics': metrics,
            'reward': reward,
            'config': isa_config
        }
    
    def export_results(self, results: Dict[str, Any], filename: str = "rl_optimization_results.json"):
        """Export optimization results"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results exported to {filename}")

def run_rl_optimization(workload_data: Dict[str, Any], episodes: int = 100):
    """Run RL-based ISA optimization"""
    optimizer = ISAOptimizer()
    results = optimizer.optimize_isa(workload_data, episodes)
    
    print("\n=== RL Optimization Results ===")
    print(f"Best Configuration: {results['best_config']}")
    print(f"Best Reward: {results['best_reward']:.2f}")
    print(f"Final Metrics: {results['final_metrics']}")
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_workload = {
        'layers': [
            {'name': 'conv1', 'type': 'Conv2d', 'avg_time': 0.089, 'percentage': 36.3, 'parameters': 864, 'flops_estimate': 430000},
            {'name': 'conv2', 'type': 'Conv2d', 'avg_time': 0.054, 'percentage': 22.0, 'parameters': 18432, 'flops_estimate': 920000},
            {'name': 'relu1', 'type': 'ReLU', 'avg_time': 0.032, 'percentage': 13.1, 'parameters': 0, 'flops_estimate': 150000},
        ]
    }
    
    results = run_rl_optimization(sample_workload, episodes=50)