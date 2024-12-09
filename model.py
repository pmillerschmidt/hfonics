import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import pretty_midi
from pathlib import Path
import logging
from typing import List, Dict


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_mask)
        return self.fc_out(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


def train_model(model, train_loader, epochs=10, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} complete, Average Loss: {avg_loss:.4f}')


class RLFineTuner:
    def __init__(self, base_model, learning_rate=1e-4):
        self.base_model = base_model
        self.policy_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, base_model.fc_out.out_features)
        )
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def get_action(self, state):
        with torch.no_grad():
            base_logits = self.base_model(state)
        policy_adjustment = self.policy_net(base_logits)
        final_logits = base_logits + policy_adjustment
        return torch.softmax(final_logits, dim=-1)

    def update(self, states, actions, rewards):
        base_logits = self.base_model(states)
        policy_adjustments = self.policy_net(base_logits)
        final_logits = base_logits + policy_adjustments

        # PPO-style update
        log_probs = F.log_softmax(final_logits, dim=-1)
        selected_log_probs = torch.sum(log_probs * actions, dim=-1)

        # Calculate loss (policy gradient with baseline)
        advantages = rewards - rewards.mean()
        policy_loss = -(selected_log_probs * advantages).mean()

        # KL penalty to prevent too much deviation from base model
        kl_div = F.kl_div(F.log_softmax(base_logits, dim=-1),
                          F.softmax(final_logits, dim=-1))

        total_loss = policy_loss + 0.01 * kl_div

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()