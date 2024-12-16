import torch
import torch.nn as nn

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
        # ppo update
        log_probs = F.log_softmax(final_logits, dim=-1)
        selected_log_probs = torch.sum(log_probs * actions, dim=-1)
        # calc loss
        advantages = rewards - rewards.mean()
        policy_loss = -(selected_log_probs * advantages).mean()
        # KL penalty to prevent deviation
        kl_div = F.kl_div(F.log_softmax(base_logits, dim=-1),
                          F.softmax(final_logits, dim=-1))
        total_loss = policy_loss + 0.01 * kl_div
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()