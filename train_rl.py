import torch
from typing import List, Dict
import logging
from pathlib import Path
from .environment import MusicEnvironment
from .agent import PPOAgent
from model.transformer import MusicTransformer


def collect_rollout(
        env: MusicEnvironment,
        agent: PPOAgent,
        max_steps: int = 1000
) -> List[Dict]:
    """Collect a single rollout"""
    rollout = []
    state = env.reset()

    for _ in range(max_steps):
        action, log_prob = agent.select_action(state)
        next_state, reward, done = env.step(action)

        rollout.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob
        })

        if done:
            break

        state = next_state

    return rollout


def train_rl(
        base_model: MusicTransformer,
        num_episodes: int = 1000,
        steps_per_episode: int = 1000,
        save_dir: Path = Path('runs/rl_finetuning')
) -> None:
    """Main RL training loop"""
    # Setup
    env = MusicEnvironment(base_model)
    agent = PPOAgent(
        state_dim=base_model.d_model,
        action_dim=base_model.fc_out.out_features
    )

    # Training loop
    for episode in range(num_episodes):
        # Collect experience
        rollout = collect_rollout(env, agent, steps_per_episode)

        # Update agent
        actor_loss, critic_loss = agent.update(rollout)

        # Log progress
        if episode % 10 == 0:
            avg_reward = sum(r['reward'] for r in rollout) / len(rollout)
            logging.info(
                f'Episode {episode}, '
                f'Avg Reward: {avg_reward:.2f}, '
                f'Actor Loss: {actor_loss:.4f}, '
                f'Critic Loss: {critic_loss:.4f}'
            )

        # Save periodically
        if episode % 100 == 0:
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'avg_reward': avg_reward
            }, save_dir / f'rl_checkpoint_{episode}.pt')