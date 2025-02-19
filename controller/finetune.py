import os
import json
import torch
import einops
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from tqdm.notebook import tqdm
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import ReplayBuffer, StaticDataset
from .configs import FinetuneConfig, TrainConfig
from .models import (
    ObservationModel,
    TransitionModel,
    RewardModel,
    PosteriorModel,
)


def collect_data(env: gym.Env, num_episodes: int):
    buffer = ReplayBuffer(
        capacity=num_episodes*env.horizon,
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    print("collecting data")
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, done)
            obs = next_obs

    return buffer


def finetune(env: gym.Env, config: FinetuneConfig):

    # prepare logging
    finetune_dir = Path(config.log_dir) / "finetune" / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(finetune_dir, exist_ok=True)
    with open(finetune_dir / "finetune_config.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=finetune_dir)

    # load training data
    train_dir = Path(config.log_dir) / "train" / config.train_id
    with open(train_dir / "train_config.json", "r") as f:
        train_config = TrainConfig(**json.load(f))

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # create datasets
    buffer = collect_data(env=env, num_episodes=config.num_episodes)
    dataset = StaticDataset(replay_buffer=buffer, chunk_length=config.chunk_length)

    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[1-config.test_size, config.test_size],
    )

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    observation_model = ObservationModel(
        state_dim=train_config.state_dim,
        observation_dim=env.observation_space.shape[0],
    ).to(device)

    transition_model = TransitionModel(
        state_dim=train_config.state_dim,
        action_dim=env.action_space.shape[0],
        min_std=train_config.min_std,
    ).to(device)

    posterior_model = PosteriorModel(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        state_dim=train_config.state_dim,
        rnn_hidden_dim=train_config.rnn_hidden_dim,
        rnn_input_dim=train_config.rnn_input_dim,
        min_std=train_config.min_std,
    ).to(device)

    # load state dicts
    observation_model.load_state_dict(torch.load(train_dir / "obs_model.pth", weights_only=True))
    transition_model.load_state_dict(torch.load(train_dir / "transition_model.pth", weights_only=True))
    posterior_model.load_state_dict(torch.load(train_dir / "posterior_model.pth", weights_only=True))
    
    # freeze all the other models and only learn a reward model
    for p in transition_model.parameters():
        p.requires_grad = False
    
    for p in observation_model.parameters():
        p.requires_grad = False

    for p in posterior_model.parameters():
        p.requires_grad = False

    posterior_model.eval()
    transition_model.eval()
    observation_model.eval()
    
    reward_model = RewardModel(
        state_dim=train_config.state_dim,
    ).to(device)

    all_params = list(reward_model.parameters())

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # main training loop
    print("training begins")
    for epoch in tqdm(range(config.num_epochs)):

        # train
        reward_model.train()
        for batch, (observations, actions, rewards) in enumerate(train_dataloader):
            observations = observations.to(device)
            observations = einops.rearrange(observations, 'b l o -> l b o')
            actions = actions.to(device)
            actions = einops.rearrange(actions, 'b l a -> l b a')
            rewards = rewards.to(device)
            rewards = einops.rearrange(rewards, 'b l r -> l b r')
            
            # prepare Tensor to maintain states sequence and rnn hidden sequence
            posterior_samples = torch.zeros(
                (config.chunk_length, config.batch_size, train_config.state_dim),
                device=device
            )
            rnn_hiddens = torch.zeros(
                (config.chunk_length, config.batch_size, train_config.rnn_hidden_dim),
                device=device
            )

            # first step is a bit different from the others
            rnn_hidden, state_posterior = posterior_model(
                prev_rnn_hidden=torch.zeros(config.batch_size, train_config.rnn_hidden_dim, device=device),
                prev_action=torch.zeros(config.batch_size, env.action_space.shape[0], device=device),
                observation=observations[0],
            )

            posterior_sample = state_posterior.rsample()

            rnn_hiddens[0] = rnn_hidden
            posterior_samples[0] = posterior_sample

            for t in range(1, config.chunk_length):
                rnn_hidden, state_posterior = posterior_model(
                    prev_rnn_hidden=rnn_hidden,
                    prev_action=actions[t-1],
                    observation=observations[t],
                )
                posterior_sample = state_posterior.rsample()
                rnn_hiddens[t] = rnn_hidden
                posterior_samples[t] = posterior_sample

            flatten_posterior_samples = posterior_samples.reshape(-1, train_config.state_dim)
            
            recon_rewards = reward_model(
                flatten_posterior_samples,
            ).reshape(config.chunk_length, config.batch_size, 1)

            reward_loss = 0.5 * mse_loss(
                recon_rewards,
                rewards,
                reduction='none'
            ).mean([0, 1]).sum()

            loss = reward_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, config.clip_grad_norm)
            optimizer.step()

            total_idx = epoch * len(train_dataloader) + batch 
            writer.add_scalar('reward loss train', loss.item(), total_idx)

        # test
        reward_model.eval()
        with torch.no_grad():
            for batch, (observations, actions, rewards) in enumerate(test_dataloader):
                observations = observations.to(device)
                observations = einops.rearrange(observations, 'b l o -> l b o')
                actions = actions.to(device)
                actions = einops.rearrange(actions, 'b l a -> l b a')
                rewards = rewards.to(device)
                rewards = einops.rearrange(rewards, 'b l r -> l b r')
                
                # prepare Tensor to maintain states sequence and rnn hidden sequence
                posterior_samples = torch.zeros(
                    (config.chunk_length, config.batch_size, train_config.state_dim),
                    device=device
                )
                rnn_hiddens = torch.zeros(
                    (config.chunk_length, config.batch_size, train_config.rnn_hidden_dim),
                    device=device
                )

                # first step is a bit different from the others
                rnn_hidden, state_posterior = posterior_model(
                    prev_rnn_hidden=torch.zeros(config.batch_size, train_config.rnn_hidden_dim, device=device),
                    prev_action=torch.zeros(config.batch_size, env.action_space.shape[0], device=device),
                    observation=observations[0],
                )

                posterior_sample = state_posterior.rsample()

                rnn_hiddens[0] = rnn_hidden
                posterior_samples[0] = posterior_sample

                for t in range(1, config.chunk_length):
                    rnn_hidden, state_posterior = posterior_model(
                        prev_rnn_hidden=rnn_hidden,
                        prev_action=actions[t-1],
                        observation=observations[t],
                    )
                    posterior_sample = state_posterior.rsample()
                    rnn_hiddens[t] = rnn_hidden
                    posterior_samples[t] = posterior_sample

                flatten_posterior_samples = posterior_samples.reshape(-1, train_config.state_dim)
                
                recon_rewards = reward_model(
                    flatten_posterior_samples,
                ).reshape(config.chunk_length, config.batch_size, 1)

                reward_loss = 0.5 * mse_loss(
                    recon_rewards,
                    rewards,
                    reduction='none'
                ).mean([0, 1]).sum()

                total_idx = epoch * len(test_dataloader) + batch 
                writer.add_scalar('reward loss test', loss.item(), total_idx)

     # save learned model parameters
    torch.save(reward_model.state_dict(), finetune_dir / "reward_model.pth")
    writer.close()
    
    return {"finetune dir": finetune_dir}   