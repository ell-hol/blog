# Building an AI-Powered Snake Game with Reinforcement Learning

![Snake Game](https://private-user-images.githubusercontent.com/21223467/363158403-6119676f-b9d2-4e6f-8dc9-baac4e054529.png)

The Snake game is a classic arcade game that has entertained millions over decades. In this article, we will explore an implementation of the Snake game using Python and PyTorch, enhanced with Reinforcement Learning (RL) to train an AI agent to master the game. We will guide you through the components of this project, including the game environment, the neural network architecture, and the reinforcement learning agent.

## Overview of the Code

### Key Libraries and Modules

1. **PyGame**: A popular library for writing video games in Python.
2. **PyTorch**: A deep learning library used to build the neural network for the RL agent.
3. **NumPy**: For numerical operations and state representation.
4. **Matplotlib**: To visualize the training progress.

### Argument Parsing

To allow flexibility in running the script either for training or inference, command-line arguments are parsed using `argparse`.

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Train or run the Snake game with RL")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--run", action="store_true", help="Run the model in inference mode")
    parser.add_argument("--weights", type=str, default="checkpoints/snake_dqn_model.pth", help="Path to the weights file")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes")
    return parser.parse_args()
```

### Snake Game Environment

The `SnakeEnv` class defines the game environment, which handles the game logic, rendering, and interaction with the AI agent. This environment uses the PyGame library to create a visual representation of the game.

#### Key Components of the Game Environment

- **Game Initialization**: Sets up the game window, colors, snake speed, and other game parameters.
- **Game State Reset (`reset`)**: Resets the game state, including the snake's position, food location, and game variables.
- **Game Step (`step`)**: Takes an action (move direction) as input, updates the game state, checks for collisions, updates the score, and returns the new state, reward, and game-over status.
- **Game Render (`render`)**: Refreshes the display to show the current game state.

Here's an excerpt from the `SnakeEnv` class showing the `reset` method:

```python
def reset(self):
    self.game_over = False
    self.x1 = self.width / 2
    self.y1 = self.height / 2
    self.x1_change = 0
    self.y1_change = 0
    self.snake_list = [(self.x1, self.y1)]
    self.length_of_snake = 1
    self.score = 0
    self.actions_taken = 0
    self.accumulated_reward = 0
    self.start_time = time.time()
    self.last_actions = deque(maxlen=5)
    self.steps = 0

    self.foodx = round(random.randrange(0, self.width - self.snake_block) / 20.0) * 20.0
    self.foody = round(random.randrange(0, self.height - self.snake_block) / 20.0) * 20.0

    return self._get_state()
```

### Reinforcement Learning Agent

The RL agent is implemented using a Recurrent Neural Network (RNN) architecture to handle the sequential nature of the Snake game. The agent learns to play the game by maximizing the rewards over multiple episodes of training.

#### RNN Architecture

The agent uses a Gated Recurrent Unit (GRU) for its neural network architecture:

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
```

The RNN takes the game state as input and outputs the Q-values corresponding to each possible action. The agent then selects the action with the highest Q-value.

#### Experience Replay and Training

The agent uses experience replay to store and reuse past experiences, improving learning stability:

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )
```

### Training the Agent

The `train_rnn` function is responsible for training the agent over a specified number of episodes. It initializes the game environment and the agent, then loops over each episode to let the agent play and learn from its experience:

```python
def train_rnn(episodes, render=False, checkpoint_interval=50, checkpoint_dir="checkpoints"):
    env = SnakeEnv()
    state_size = 12  # Size of the state returned by SnakeEnv (increased by 1 for time)
    action_size = 4  # Number of possible actions
    agent = RNNAgent(state_size, action_size)

    scores = []
    accumulated_rewards = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for e in range(episodes):
        state = env.reset()
        score = 0
        accumulated_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            score += reward
            accumulated_reward = info["accumulated_reward"]
            agent.train()

            if render:
                env.render()

        if e % agent.target_update == 0:
            agent.update_target_model()

        scores.append(score)
        accumulated_rewards.append(accumulated_reward)
        print(f"Episode: {e+1}/{episodes}, Score: {score:.2f}, Accumulated Reward: {accumulated_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        if (e + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{e+1}.pth")
            torch.save(agent.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {e+1}")

    torch.save(agent.model.state_dict(), os.path.join(checkpoint_dir, "snake_dqn_model.pth"))
    env.close()
    return scores, accumulated_rewards, agent
```

### Running the Game

The script can be executed in two modes:

1. **Training Mode**: Trains the RL agent to learn to play the Snake game.
2. **Inference Mode**: Runs the trained agent to play the game autonomously.

To run the script, use the following command:

```sh
python snake_game_rl.py --train --episodes 2000
```

Or for inference:

```sh
python snake_game_rl.py --run --weights checkpoints/snake_dqn_model.pth
```

### Visualizing the Results

After training, the script plots the agent's performance over time:

```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(scores)), scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("DQN Training Progress - Scores")

plt.subplot(1, 2, 2)
plt.plot(range(len(accumulated_rewards)), accumulated_rewards)
plt.xlabel("Episode")
plt.ylabel("Accumulated Reward")
plt.title("DQN Training Progress - Accumulated Rewards")

plt.tight_layout()
plt.show()
```

## Conclusion

This project showcases how Reinforcement Learning can be applied to a classic game like Snake to create an AI agent that learns to play the game effectively. By leveraging the power of PyTorch, we implemented a neural network-based agent that uses an RNN for sequential decision-making, demonstrating the potential of machine learning in game development.

Feel free to explore the code further, tune the hyperparameters, and experiment with different neural network architectures to enhance the agent's performance!
