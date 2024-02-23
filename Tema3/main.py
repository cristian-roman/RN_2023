import random
import torch
import torch.nn as nn
import torch.optim as optim
import flappy_bird_gymnasium
import gymnasium

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

        self.fc2 = nn.Linear(256, 128)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

        self.fc3 = nn.Linear(128, output_size)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Deep Q-Learning parameters
learning_rate = 0.0001
gamma = 0.9
epsilon = 1.0
decay = 0.9995

env = gymnasium.make("FlappyBird-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = QNetwork(state_size, action_size)

q_network.train()
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

num_episodes = 10000

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0

    while True:
        if random.uniform(0, 1) < epsilon:
            if random.uniform(0, 1) < 0.85:
                action = 0
            else:
                action = 1
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

        next_obs, reward, terminated, _, info = env.step(action)

        if next_obs[9] <= 0:
            reward = -1.0
            terminated = True

        with torch.no_grad():
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            target_q_values = q_network(next_state_tensor)
            max_next_q_value = torch.max(target_q_values).item()
            target_q_value = torch.tensor([reward + gamma * max_next_q_value], dtype=torch.float32)

        optimizer.zero_grad()
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = q_network(state_tensor)
        loss = loss_function(q_values[0][action], target_q_value)
        loss.backward()
        optimizer.step()

        total_reward += reward
        obs = next_obs

        if terminated:
            break
    epsilon *= decay

print("Training complete.")
env.close()

# save the model
torch.save(q_network.state_dict(), 'flappy_bird_model.pth')

q_network = QNetwork(12, 2)
q_network.load_state_dict(torch.load('flappy_bird_model.pth'))
env = gymnasium.make("FlappyBird-v0", render_mode="human")
q_network.eval()

while True:
    obs, _ = env.reset()
    total_reward = 0

    while True:
        with torch.no_grad():
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        next_obs, reward, terminated, _, info = env.step(action)

        total_reward += reward
        obs = next_obs

        if terminated:
            break
    print("Total reward: {}".format(total_reward))
