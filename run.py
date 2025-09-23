import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque

# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_features = self.shared_net(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

# --- Reference Buffer for Storing Best Trajectories ---
class ReferenceBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.buffer = [] # Stores (cumulative_reward, trajectory) tuples
        self.min_reward = -float('inf')

    def add(self, trajectory):
        """Adds a trajectory if it's better than the worst in the buffer."""
        cumulative_reward = sum([t[2] for t in trajectory]) # sum of extrinsic rewards
        if len(self.buffer) < self.capacity or cumulative_reward > self.min_reward:
            self.buffer.append((cumulative_reward, trajectory))
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            if len(self.buffer) > self.capacity:
                self.buffer.pop()
            self.min_reward = self.buffer[-1][0]

    def get_states(self, batch_size=32):
        """Returns a random batch of states from the best trajectories."""
        if not self.buffer:
            return []
        
        # Unpack all states from all trajectories in the buffer
        all_states = [step[0] for _, traj in self.buffer for step in traj]
        
        # Sample randomly
        indices = np.random.choice(len(all_states), size=batch_size, replace=True)
        return [all_states[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

# --- Meta-Controller (UCB1 Bandit) for Objective Selection ---
class MetaController:
    def __init__(self, num_objectives):
        self.num_objectives = num_objectives
        self.counts = np.zeros(num_objectives)
        self.values = np.zeros(num_objectives)
        self.total_count = 0

    def select_objective(self):
        """Selects an objective using the UCB1 formula."""
        self.total_count += 1
        # Play each arm once first
        if self.total_count <= self.num_objectives:
            return self.total_count - 1
        
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_count) / self.counts)
        return np.argmax(ucb_values)

    def update(self, objective_idx, reward_improvement):
        """Updates the value and count for the chosen objective."""
        self.counts[objective_idx] += 1
        n = self.counts[objective_idx]
        old_value = self.values[objective_idx]
        # Update value using incremental mean formula
        self.values[objective_idx] = old_value + (reward_improvement - old_value) / n

# --- Base Class for Intrinsic Reward Modules ---
class IntrinsicRewardModule:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def compute_reward(self, state, action, next_state, reference_buffer, actor_critic_model):
        raise NotImplementedError

class RNDModel(nn.Module): # For Objective 1
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.target = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        # Freeze target network parameters
        for param in self.target.parameters():
            param.requires_grad = False
    
    def forward(self, state):
        return torch.pow(self.predictor(state) - self.target(state), 2).mean()

class DynamicsModel(nn.Module): # For Objective 3 & 9
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # This model learns to predict next_state from (state, action)
        self.model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, state_dim))
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1))



class Objective1_Exploration(IntrinsicRewardModule):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.rnd_model = RNDModel(state_dim)
        # Note: The predictor part of rnd_model needs to be trained on states the agent visits.

    def compute_reward(self, state, **kwargs):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Reward is the novelty score (prediction error)
        return self.rnd_model(state_tensor).item()

class Objective2_Exploitation(IntrinsicRewardModule):
    def compute_reward(self, **kwargs):
        # No intrinsic reward, focuses solely on the environment's extrinsic reward.
        return 0.0

class Objective3_Extrapolation(IntrinsicRewardModule):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        # Note: This model needs to be trained on data from the reference buffer.
        
    def compute_reward(self, state, action, next_state, **kwargs):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        
        pred_next_state = self.dynamics_model(state_t, action_t)
        error = torch.pow(pred_next_state - next_state_t, 2).mean().item()
        # Reward is given for behaving in a way that is "predictable" by the baseline dynamics.
        return -error

class Objective4_Spreading(IntrinsicRewardModule):
    def compute_reward(self, state, reference_buffer, **kwargs):
        baseline_states = reference_buffer.get_states(batch_size=32)
        if len(baseline_states) < 2:
            return 0.0
        
        state_vec = np.array(state)
        baseline_vecs = np.array(baseline_states)
        
        # Reward is proportional to the distance to the mean of the baseline states.
        # This encourages the agent to explore the periphery of the known good states.
        mean_baseline = np.mean(baseline_vecs, axis=0)
        distance_to_mean = np.linalg.norm(state_vec - mean_baseline)
        return distance_to_mean

class Objective5_DataGathering(IntrinsicRewardModule):
    def compute_reward(self, state, actor_critic_model, **kwargs):
        # Reward is the policy's entropy at the current state.
        # High entropy means the policy is uncertain, so acting here is informative.

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, _ = actor_critic_model(state_t)
            dist = Categorical(logits=logits)
        return dist.entropy().item()

class Objective6_PairMinimization(IntrinsicRewardModule):
    def compute_reward(self, state, reference_buffer, **kwargs):
        s_a_list = reference_buffer.get_states(batch_size=1)
        s_b_list = reference_buffer.get_states(batch_size=1)
        if not s_a_list or not s_b_list:
            return 0.0
        
        s_a = np.array(s_a_list[0])
        s_b = np.array(s_b_list[0])
        s_t = np.array(state)
        
        d_ab = np.linalg.norm(s_a - s_b)
        d_at = np.linalg.norm(s_a - s_t)
        d_tb = np.linalg.norm(s_t - s_b)
        
        return d_ab - (d_at + d_tb)

class Objective7_MetricSimilarity(IntrinsicRewardModule):
    def compute_reward(self, state, reference_buffer, **kwargs):
        baseline_states = reference_buffer.get_states(batch_size=32)
        if not baseline_states:
            return 0.0
        
        state_vec = np.array(state)
        baseline_vecs = np.array(baseline_states)
        
        distances = np.linalg.norm(baseline_vecs - state_vec, axis=1)
        min_dist = np.min(distances)
        
        return np.exp(-min_dist**2)

class Objective8_InterpolationGoal(IntrinsicRewardModule):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.goal_state = None

    def new_episode(self, reference_buffer):
        s_a_list = reference_buffer.get_states(batch_size=1)
        s_b_list = reference_buffer.get_states(batch_size=1)
        if s_a_list and s_b_list:
            alpha = np.random.rand() # Random interpolation factor
            self.goal_state = alpha * np.array(s_a_list[0]) + (1 - alpha) * np.array(s_b_list[0])
        else:
            self.goal_state = None
            
    def compute_reward(self, state, **kwargs):
        if self.goal_state is None:
            return 0.0
        # Provide a goal-reaching reward for the interpolated state.
        dist = np.linalg.norm(np.array(state) - self.goal_state)
        return np.exp(-dist**2)

class Objective9_IntegratedExploration(IntrinsicRewardModule):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.rnd_model = RNDModel(state_dim)
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        # Both models would need to be trained.
        
    def compute_reward(self, state, action, next_state, **kwargs):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

        # Reward for novelty (high RND error)
        novelty_reward = self.rnd_model(state_t).item()
        
        # Reward for being in a state where the world model is uncertain (high dynamics error)
        pred_next_state = self.dynamics_model(state_t, action_t)
        dynamics_error = torch.pow(pred_next_state - next_state_t, 2).mean().item()
        
        # Combine the two rewards.
        return novelty_reward + dynamics_error

class Objective10_MetricMatching(IntrinsicRewardModule):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.goal_state = None

    def new_episode(self, reference_buffer):
        goal_states = reference_buffer.get_states(batch_size=1)
        if goal_states:
            self.goal_state = goal_states[0]
        else:
            self.goal_state = None

    def compute_reward(self, state, **kwargs):
        if self.goal_state is None:
            return 0.0
        # Goal-reaching reward for a state sampled directly from the baseline.
        dist = np.linalg.norm(np.array(state) - np.array(self.goal_state))
        return np.exp(-dist**2)

class Objective11_SpreadingFromBaseline(IntrinsicRewardModule):
    def compute_reward(self, state, reference_buffer, **kwargs):
        baseline_states = reference_buffer.get_states(batch_size=32)
        if not baseline_states:
            return 0.0
        
        state_vec = np.array(state)
        baseline_vecs = np.array(baseline_states)
        
        distances = np.linalg.norm(baseline_vecs - state_vec, axis=1)
        min_dist = np.min(distances)
        
        return min_dist

class Objective12_ExpandingBaseline(IntrinsicRewardModule):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        
    def compute_reward(self, state, action, next_state, reference_buffer, **kwargs):

        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        pred_next_state = self.dynamics_model(state_t, action_t)
        dynamics_error = torch.pow(pred_next_state - next_state_t, 2).mean().item()
        plausibility_reward = -dynamics_error
        
        novelty_reward = Objective11_SpreadingFromBaseline.compute_reward(
            self, state, reference_buffer, **kwargs
        )
        
        return plausibility_reward + 0.1 * novelty_reward


class StubbedObjective(IntrinsicRewardModule):
    def compute_reward(self, **kwargs): return 0.0


class CAPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4,
                 eps_clip=0.2, intrinsic_beta=0.1):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.intrinsic_beta = intrinsic_beta

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        
        self.reference_buffer = ReferenceBuffer(capacity=50)
        self.meta_controller = MetaController(num_objectives=12)
        
        self.objective_modules = [
            Objective1_Exploration(state_dim, action_dim),
            Objective2_Exploitation(state_dim, action_dim),
            Objective3_Extrapolation(state_dim, action_dim),
            StubbedObjective(state_dim, action_dim), # 4
            Objective5_DataGathering(state_dim, action_dim),
            StubbedObjective(state_dim, action_dim), # 6
            Objective7_MetricSimilarity(state_dim, action_dim),
            StubbedObjective(state_dim, action_dim), # 8
            StubbedObjective(state_dim, action_dim), # 9
            Objective10_MetricMatching(state_dim, action_dim),
            StubbedObjective(state_dim, action_dim), # 11
            StubbedObjective(state_dim, action_dim)  # 12
        ]
        
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, _ = self.policy_old(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.tensor(memory['states'], dtype=torch.float32), 1)
        old_actions = torch.squeeze(torch.tensor(memory['actions'], dtype=torch.int64), 1)
        old_logprobs = torch.squeeze(torch.tensor(memory['logprobs'], dtype=torch.float32), 1)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            action_logits, state_values = self.policy(old_states)
            dist = Categorical(logits=action_logits)
            logprobs = dist.log_prob(old_actions)
            state_values = torch.squeeze(state_values)
            dist_entropy = dist.entropy()

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    max_ep_len = 500
    max_training_timesteps = int(3e5)
    update_timestep = 2000 # update policy every n timesteps
    
    agent = CAPOAgent(state_dim, action_dim)

    time_step = 0
    i_episode = 0
    last_avg_reward = 0

    while time_step <= max_training_timesteps:

        objective_idx = agent.meta_controller.select_objective()
        print(f"Iteration starting at timestep {time_step}. Using Objective: {objective_idx+1}")
        
        if hasattr(agent.objective_modules[objective_idx], 'new_episode'):
            agent.objective_modules[objective_idx].new_episode(agent.reference_buffer)

        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
        current_ep_rewards = []
        
        while len(memory['states']) < update_timestep:
            state, _ = env.reset()
            current_episode_trajectory = []
            ep_extrinsic_reward = 0
            
            for t in range(1, max_ep_len + 1):
                action, log_prob = agent.select_action(state)
                next_state, extrinsic_reward, done, truncated, _ = env.step(action)

                # Compute intrinsic reward
                intrinsic_reward = agent.objective_modules[objective_idx].compute_reward(
                    state=state, 
                    action=action, 
                    next_state=next_state,
                    reference_buffer=agent.reference_buffer,
                    actor_critic_model=agent.policy_old
                )
                
                # Combine rewards
                total_reward = extrinsic_reward + agent.intrinsic_beta * intrinsic_reward

                # Saving data
                current_episode_trajectory.append((state, action, extrinsic_reward, next_state, done or truncated))
                memory['states'].append(state)
                memory['actions'].append(action)
                memory['logprobs'].append(log_prob)
                memory['rewards'].append(total_reward)
                memory['is_terminals'].append(done or truncated)
                
                state = next_state
                ep_extrinsic_reward += extrinsic_reward
                
                if done or truncated:
                    break
            
            agent.reference_buffer.add(current_episode_trajectory)
            current_ep_rewards.append(ep_extrinsic_reward)
            i_episode += 1

        time_step += len(memory['states'])
        
        # --- UPDATE POLICY ---
        agent.update(memory)
        
        # --- UPDATE META-CONTROLLER ---
        current_avg_reward = np.mean(current_ep_rewards)
        reward_improvement = current_avg_reward - last_avg_reward
        agent.meta_controller.update(objective_idx, reward_improvement)
        last_avg_reward = current_avg_reward
        
        print(f"Episode {i_episode}, Timestep {time_step}, Avg Extrinsic Reward: {current_avg_reward:.2f}")

if __name__ == '__main__':
    main()