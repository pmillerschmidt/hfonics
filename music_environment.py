import torch
from model import MusicTransformer

class MusicEnvironment:
    def __init__(self,
                 base_model: MusicTransformer,
                 sequence_length: int = 512):
        self.base_model = base_model
        self.sequence_length = sequence_length
        self.current_sequence = None
        self.position = 0

    def reset(self):
        self.current_sequence = torch.zeros(1, self.sequence_length, dtype=torch.long)
        self.position = 0
        return self.current_sequence

    def step(self, action):
        # Apply action to modify base model's predictions
        with torch.no_grad():
            base_logits = self.base_model(self.current_sequence)
            modified_logits = base_logits + action
            token = torch.multinomial(torch.softmax(modified_logits, dim=-1), 1)
        # Update sequence
        self.current_sequence[0, self.position] = token
        self.position += 1
        done = self.position >= self.sequence_length
        return self.current_sequence, 0.0, done  # Reward from human feedback