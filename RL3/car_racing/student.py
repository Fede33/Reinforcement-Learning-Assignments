import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

INPUT_CHANNELS = 3   # for RBG images

class Policy(nn.Module):
    #continuous = False 

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()

        self.device = device

        # Discrete Version
        self.continuous = False  
        self.log_prob = None
        self.entropy = None

        # Environment 
        self.env = gym.make(
            'CarRacing-v2',
            continuous=self.continuous,
            render_mode="rgb_array"
        )

        self.rewards_history = []   # Total rewards per episode
        self.loss_history = []      # Total loss per episode

        # Action space
        if not self.continuous:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.action_high = torch.from_numpy(self.env.action_space.high).to(self.device)
            self.action_low = torch.from_numpy(self.env.action_space.low).to(self.device)

        
        #  Network architecture 
        self.shared = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True)
        )

        self.actor_mean = nn.Linear(256, self.action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim))  # only if continuous=True
        self.activation_actor_std = nn.Softplus()
        self.critic = nn.Linear(256, 1)

        # Init weughts conv
        for layer in self.shared:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Init linear in shared
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Init actor and critic
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        if self.actor_mean.bias is not None:
            nn.init.zeros_(self.actor_mean.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            nn.init.zeros_(self.critic.bias)

    
        self.to(self.device)

        #  PPO hyperparameters
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.max_episodes = 7500
        self.ppo_update_epochs = 5
        self.batch_size = 64
        self.max_timesteps = self.batch_size * 4
        self.skip_frames = 4

    def forward(self, x):
        # TODO
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.array(x, copy=False))
        x = x.to(self.device, dtype=torch.float32)

        # convert to NCHW
        if x.dim() == 3:
            # [H, W, C] -> [1, C, H, W]
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4 and x.shape[-1] == INPUT_CHANNELS:
            # [N, H, W, C] -> [N, C, H, W]
            x = x.permute(0, 3, 1, 2)

        # Normalize images
        x = x / 255.0

        x = self.shared(x)
        action_logits = self.actor_mean(x)
        state_value = self.critic(x).squeeze(-1)  # [N]

        if not self.continuous:
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs, state_value
        else:
            action_mean = action_logits
            return action_mean, state_value
    
    def act(self, state):
        # TODO
        # state: numpy array (H, W, C)
        with torch.no_grad():
            mean_or_probs, _ = self(state)

            if not self.continuous:
                action_probs = mean_or_probs
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()              # [1]
            else:
                action_mean = mean_or_probs
                action_log_std = self.actor_log_std.expand_as(action_mean)
                action_std = self.activation_actor_std(action_log_std)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()              # [1, action_dim]
                action = torch.clamp(action, self.action_low, self.action_high)

            log_prob = dist.log_prob(action)
            if self.continuous:
                log_prob = log_prob.sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
            else:
                entropy = dist.entropy()

        
        self.log_prob = log_prob
        self.entropy = entropy

        if self.continuous:
            action_np = action.detach().cpu().numpy().astype(np.float32)
            action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)
            return action_np[0]
        else:
            action_np = action.detach().cpu().numpy().astype(np.int64)
            return int(action_np[0])
        

# GAE / returns
    def compute_advantages(self, rewards, dones, values):
        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[i])
            advantages.insert(0, gae)
            next_value = values[i]
            returns.insert(0, gae + values[i])

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        return returns, advantages
        

    def train(self):
        # TODO
        print(
            f"Training started: max_episodes = {self.max_episodes}, "
            f"max_timesteps = {self.max_timesteps}, "
            f"ppo_update_epochs = {self.ppo_update_epochs}, "
            f"continuous = {self.continuous}\n"
        )

        all_episode_rewards = []
        all_losses = []
        all_actor_losses = []
        all_critic_losses = []
        all_entropies = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            memory = [[], [], [], [], []]  # states, actions, rewards, log_probs, dones
            total_reward = 0.0

            for t in range(self.max_timesteps):
                action = self.act(state)          # update self.log_prob, self.entropy
                log_prob = self.log_prob
                entropy = self.entropy           

                four_frames_reward = 0.0
                for _ in range(self.skip_frames):
                    next_state, reward, done, _, _ = self.env.step(action)
                    reward = np.clip(reward, a_min=None, a_max=1.0)
                    four_frames_reward += reward
                    if done:
                        break

                avg_reward = four_frames_reward / self.skip_frames
                total_reward += avg_reward

                # Store transition
                memory[0].append(state)
                memory[1].append(action)
                memory[2].append(avg_reward)
                memory[3].append(log_prob.item())
                memory[4].append(float(done))

                state = next_state
                if done:
                    break

            all_episode_rewards.append(total_reward)

            if episode % 100 == 0:
                print(f"\tEpisode {episode}: Reward = {total_reward:.2f}")

            # PPO update
            states = np.array(memory[0])  # [T, H, W, C]

            # States -> tensor NCHW
            states = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(self.device)

            actions_np = np.array(memory[1])
            if self.continuous:
                actions = torch.from_numpy(actions_np).float().to(self.device)
            else:
                # LongTensor for discrete actions
                actions = torch.from_numpy(actions_np).long().to(self.device)

            rewards = torch.tensor(memory[2], dtype=torch.float32, device=self.device)
            log_probs_old = torch.tensor(memory[3], dtype=torch.float32, device=self.device)
            dones = torch.tensor(memory[4], dtype=torch.float32, device=self.device)

            # V(s) values
            _, values = self(states)
            values = values.detach()  

            returns, advantages = self.compute_advantages(rewards, dones, values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self.rewards_history.append(total_reward)

            episode_loss_total = 0.0
            episode_actor_loss_total = 0.0
            episode_critic_loss_total = 0.0
            episode_entropy_total = 0.0

            T = states.size(0)
            num_minibatches = max(1, (T + self.batch_size - 1) // self.batch_size)
            num_updates = self.ppo_update_epochs * num_minibatches

            for epoch in range(self.ppo_update_epochs):
                permutation = torch.randperm(T, device=self.device)

                for i in range(0, T, self.batch_size):
                    indices = permutation[i:i + self.batch_size]

                    batch_states = states[indices]
                    batch_actions = actions[indices]
                    batch_returns = returns[indices]
                    batch_advantages = advantages[indices]
                    batch_log_probs_old = log_probs_old[indices]

                    # Forward
                    action_out, batch_values = self(batch_states)

                    if not self.continuous:
                        dist = torch.distributions.Categorical(action_out)
                        batch_log_probs = dist.log_prob(batch_actions)
                        batch_entropy = dist.entropy()
                    else:
                        action_mean = action_out
                        action_log_std = self.actor_log_std.expand_as(action_mean)
                        action_std = self.activation_actor_std(action_log_std)
                        dist = torch.distributions.Normal(action_mean, action_std)
                        batch_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                        batch_entropy = dist.entropy().sum(dim=-1)

                    ratios = torch.exp(batch_log_probs - batch_log_probs_old)

                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    actor_loss = torch.min(surr1, surr2).mean()

                    batch_values = batch_values.squeeze()
                    critic_loss = F.mse_loss(batch_values, batch_returns)

                    entropy = batch_entropy.mean()

                    loss = -actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    episode_loss_total += loss.item()
                    episode_actor_loss_total += actor_loss.item()
                    episode_critic_loss_total += critic_loss.item()
                    episode_entropy_total += entropy.item()

            all_losses.append(episode_loss_total / num_updates)
            all_actor_losses.append(episode_actor_loss_total / num_updates)
            all_critic_losses.append(episode_critic_loss_total / num_updates)
            all_entropies.append(episode_entropy_total / num_updates)

            if episode % 100 == 0:
                print(f"\tEpisode Total loss = {episode_loss_total / num_updates:.4f}")
                print(f"\t\tSurrogate Objective = {episode_actor_loss_total / num_updates:.4f}")
                print(f"\t\tCritic Loss Term = {episode_critic_loss_total / num_updates * self.value_coef:.4f}")
                print(f"\t\tEntropy Bonus Term = {episode_entropy_total / num_updates * self.entropy_coef:.4f}\n")

            if total_reward >= 900:
                print("Environment solved!")
                break

        
# save and load
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
