import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
import ale_py
import shimmy
import cv2
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time

def preprocess_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        # CNN 부분 (초기에는 CPU 상의 파라미터)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 아직 모델은 CPU에 있으므로, CPU 텐서로 shape만 계산
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # FC 레이어 부분
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def _get_conv_out_size(self, shape):
        """
        더미 텐서를 CPU로 만들어 conv를 통과시키고 출력 shape만 계산.
        모델은 아직 GPU로 옮기기 전이므로, CPU 텐서로 해도 문제 없음.
        """
        dummy = torch.zeros(1, *shape)  # CPU 텐서
        o = self.conv(dummy)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        x는 이미 .to(device) 되어 있다고 가정.
        따라서 여기서 굳이 x.to(...)를 다시 할 필요는 없음.
        """
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, state_shape, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=buffer_size)
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Cuda Is Available?", torch.cuda.is_available())

        # 메인 네트워크와 타깃 네트워크를 먼저 CPU에서 만들고
        self.model = DQN(state_shape, action_dim)
        self.target_model = DQN(state_shape, action_dim)
        
        # .to(self.device)로 이동 (GPU가 있으면 GPU에)
        self.model.to(self.device)
        self.target_model.to(self.device)
        
        # 초기 가중치 동기화
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 기타 파라미터
        self.target_update_freq = 400
        self.steps = 0
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        """
        state shape: (4, 84, 84)
        텐서를 (1, 4, 84, 84)로 만든 뒤 GPU/CPU로 옮겨 inference.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        
        with torch.no_grad():
            q_values = self.model(state_t)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # 랜덤 샘플링
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Tensor 변환
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q(s, a)
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q_target = r + gamma * max Q(s', a')
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # MSE Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class FrameStack:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def reset(self, frame):
        self.frames.clear()
        for _ in range(self.n_frames):
            self.frames.append(frame)
        return self.get_state()
    
    def step(self, frame):
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        return np.array(self.frames)

def main():
    # Breakout 환경 생성
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    
    # 실시간 플롯 설정
    plt.ion()  
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    rewards_history = []
    epsilons_history = []
    episodes_history = []
    
    frame_stack = FrameStack(4)
    state_shape = (4, 84, 84)
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_shape=state_shape,
        action_dim=action_dim,
        lr=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        buffer_size=100000
    )

    batch_size = 64
    num_episodes = 5000
    
    def update_plot():
        ax1.clear()
        ax2.clear()
        
        # 에피소드별 보상 그래프 (파란선)
        ax1.plot(episodes_history, rewards_history, 'b-', label='Episode Reward')
        
        # 전체 평균 보상 (빨간선)
        # i번째 에피소드까지의 평균을 구해서 리스트로 저장
        mean_rewards = [np.mean(rewards_history[:i]) for i in range(1, len(rewards_history)+1)]
        ax1.plot(episodes_history, mean_rewards, 'r-', label='Mean Reward')
        
        ax1.set_title('Total Reward per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()  # 범례
        
        # Epsilon 그래프
        ax2.plot(episodes_history, epsilons_history, 'g-')
        ax2.set_title('Epsilon per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.01)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        state = frame_stack.reset(state)
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = frame_stack.step(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay(batch_size)
            time.sleep(0.01)  # 플레이 속도 조절 (원하면 제거)
        
        agent.update_epsilon()
        
        # 기록
        rewards_history.append(total_reward)
        epsilons_history.append(agent.epsilon)
        episodes_history.append(episode + 1)
        
        # 500 에피소드마다 모델 저장
        if (episode + 1) % 500 == 0:
            torch.save(agent.model.state_dict(), f"breakout_dqn_{episode+1}.pth")
            print(f"Model saved at episode {episode + 1}")
        
        # GPU 메모리 사용량 출력 (참고용)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            gpu_memory = 0.0
        
        print(f"에피소드: {episode + 1}, 총 보상: {total_reward}, "
              f"입실론: {agent.epsilon:.4f}, GPU 메모리 사용량: {gpu_memory:.2f}MB")
        
        update_plot()
        
    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
