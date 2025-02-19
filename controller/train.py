import os
import json
import torch
import einops
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from tqdm.notebook import tqdm
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import ReplayBuffer, StaticDataset
from .configs import TrainConfig
from .models import (
    ObservationModel,
    TransitionModel,
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


def train(env: gym.Env, config: TrainConfig):

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=log_dir)

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
        batch_sampler=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    observation_model = ObservationModel(
        state_dim=config.state_dim,
        observation_dim=env.observation_space.shape[0],
    ).to(device)

    transition_model = TransitionModel(
        state_dim=config.state_dim,
        action_dim=env.action_space.shape[0],
        min_std=config.min_std,
    ).to(device)

    posterior_model = PosteriorModel(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        state_dim=config.state_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_input_dim=config.rnn_input_dim,
        min_std=config.min_std,
    ).to(device)

    all_params = (
        list(observation_model.parameters()) +
        list(transition_model.parameters()) +
        list(posterior_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # main training loop
    for epoch in range(config.num_epochs):

        # train
        posterior_model.train()
        transition_model.train()
        observation_model.train()

        for batch, (observations, actions, rewards) in enumerate(train_dataloader):
            observations = observations.to(device)
            observations = einops.rearrange(observations, 'b l o -> l b o')
            actions = actions.to(device)
            actions = einops.rearrange(actions, 'b l a -> l b a')
            rewards = rewards.to(device)
            rewards = einops.rearrange(rewards, 'b l r -> l b r')
            
            # prepare Tensor to maintain states sequence and rnn hidden sequence
            posterior_samples = torch.zeros(
                (config.chunk_length, config.batch_size, config.state_dim),
                device=device
            )
            rnn_hiddens = torch.zeros(
                (config.chunk_length, config.batch_size, config.rnn_hidden_dim),
                device=device
            )

            total_kl_loss = 0

            # first step is a bit different from the others
            rnn_hidden, state_posterior = posterior_model(
                prev_rnn_hidden=torch.zeros(config.batch_size, config.rnn_hidden_dim, device=device),
                prev_action=torch.zeros(config.batch_size, env.action_space.shape[0], device=device),
                observation=observations[0],
            )
            state_prior = torch.distributions.Normal(
                torch.zeros((config.batch_size, config.state_dim), device=device),
                torch.ones((config.batch_size, config.state_dim), device=device),
            )
            # kl divergence between prior and posterior
            kl_loss = kl_divergence(state_posterior, state_prior).sum(dim=1)
            total_kl_loss += kl_loss.clamp(min=config.free_nats).mean()

            posterior_sample = state_posterior.rsample()

            rnn_hiddens[0] = rnn_hidden
            posterior_samples[0] = posterior_sample

            for t in range(1, config.chunk_length):
                rnn_hidden, state_posterior = posterior_model(
                    prev_rnn_hidden=rnn_hidden,
                    prev_action=actions[t-1],
                    observation=observations[t],
                )
                state_prior = transition_model(
                    prev_state=posterior_sample,
                    prev_action=actions[t-1],
                )

                kl_loss = kl_divergence(state_posterior, state_prior).sum(dim=1)
                total_kl_loss += kl_loss.clamp(min=config.free_nats).mean()

                posterior_sample = state_posterior.rsample()

                rnn_hiddens[t] = rnn_hidden
                posterior_samples[t] = posterior_sample

            total_kl_loss /= config.chunk_length

            flatten_posterior_samples = posterior_samples.reshape(-1, config.state_dim)
            
            recon_observations = observation_model(
                flatten_posterior_samples,
            ).reshape(config.chunk_length, config.batch_size, env.observation_space.shape[0])

            obs_loss = 0.5 * mse_loss(
                recon_observations,
                observations,
                reduction='none'
            ).mean([0, 1]).sum()

            loss = config.kl_beta * total_kl_loss + obs_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, config.clip_grad_norm)
            optimizer.step()

            total_idx = epoch * len(train_dataloader) + batch 
            writer.add_scalar('overall loss train', loss.item(), total_idx)
            writer.add_scalar('kl loss train', total_kl_loss.item(), total_idx)
            writer.add_scalar('obs loss train', obs_loss.item(), total_idx)

        # test
        posterior_model.eval()
        transition_model.eval()
        observation_model.eval()

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
                    (config.chunk_length, config.batch_size, config.state_dim),
                    device=device
                )
                rnn_hiddens = torch.zeros(
                    (config.chunk_length, config.batch_size, config.rnn_hidden_dim),
                    device=device
                )

                total_kl_loss = 0

                # first step is a bit different from the others
                rnn_hidden, state_posterior = posterior_model(
                    prev_rnn_hidden=torch.zeros(config.batch_size, config.rnn_hidden_dim, device=device),
                    prev_action=torch.zeros(config.batch_size, env.action_space.shape[0], device=device),
                    observation=observations[0],
                )
                state_prior = torch.distributions.Normal(
                    torch.zeros((config.batch_size, config.state_dim), device=device),
                    torch.ones((config.batch_size, config.state_dim), device=device),
                )
                # kl divergence between prior and posterior
                kl_loss = kl_divergence(state_posterior, state_prior).sum(dim=1)
                total_kl_loss += kl_loss.clamp(min=config.free_nats).mean()

                posterior_sample = state_posterior.rsample()

                rnn_hiddens[0] = rnn_hidden
                posterior_samples[0] = posterior_sample

                for t in range(1, config.chunk_length):
                    rnn_hidden, state_posterior = posterior_model(
                        prev_rnn_hidden=rnn_hidden,
                        prev_action=actions[t-1],
                        observation=observations[t],
                    )
                    state_prior = transition_model(
                        prev_state=posterior_sample,
                        prev_action=actions[t-1],
                    )

                    kl_loss = kl_divergence(state_posterior, state_prior).sum(dim=1)
                    total_kl_loss += kl_loss.clamp(min=config.free_nats).mean()

                    posterior_sample = state_posterior.rsample()

                    rnn_hiddens[t] = rnn_hidden
                    posterior_samples[t] = posterior_sample

                total_kl_loss /= config.chunk_length

                flatten_posterior_samples = posterior_samples.reshape(-1, config.state_dim)
                
                recon_observations = observation_model(
                    flatten_posterior_samples,
                ).reshape(config.chunk_length, config.batch_size, env.observation_space.shape[0])

                obs_loss = 0.5 * mse_loss(
                    recon_observations,
                    observations,
                    reduction='none'
                ).mean([0, 1]).sum()

                total_idx = epoch * len(test_dataloader) + batch 
                writer.add_scalar('overall loss test', loss.item(), total_idx)
                writer.add_scalar('kl loss test', total_kl_loss.item(), total_idx)
                writer.add_scalar('obs loss test', obs_loss.item(), total_idx)
  

     # save learned model parameters
    torch.save(observation_model.state_dict(), log_dir / "obs_model.pth")
    torch.save(transition_model.state_dict(), log_dir / "transition_model.pth")
    torch.save(posterior_model.state_dict(), log_dir / "posterior_model.pth")
    writer.close()
    
    return {"model_dir": log_dir}   