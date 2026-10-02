import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from vsn_env import DynamicVSNEnv

# Deep Q-Network Architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Training Loop
def train():
    env = DynamicVSNEnv(num_vehicles=50, max_steps=100)
    state_dim = 5
    action_dim = 3
    
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = deque(maxlen=20000)
    
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    episodes = 200

    print("--- Starting DRL Social-Aware Offloading Training ---")
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_t).argmax().item()
            
            next_state, reward, done, info = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # Replay Training Step
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)
                
                s_b = torch.FloatTensor(np.array(s_b))
                a_b = torch.LongTensor(a_b).unsqueeze(1)
                r_b = torch.FloatTensor(r_b)
                ns_b = torch.FloatTensor(np.array(ns_b))
                d_b = torch.FloatTensor(d_b)
                
                q_vals = policy_net(s_b).gather(1, a_b).squeeze()
                next_q_vals = target_net(ns_b).max(1)[0]
                target_q = r_b + gamma * next_q_vals * (1 - d_b)
                
                loss = nn.MSELoss()(q_vals, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if ep % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {ep}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")

    # Save trained model weight
    torch.save(policy_net.state_dict(), "drl_vsn_model.pth")
    print("Model saved successfully as 'drl_vsn_model.pth'")

if __name__ == "__main__":
    train()