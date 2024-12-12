import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
import logging

class EnhancedGenerator:
    def __init__(self, 
                 model,
                 dataset,
                 device: str = 'cpu',
                 max_consecutive_rests: int = 8,
                 min_note_duration: int = 120,  # in ticks
                 max_note_duration: int = 1920,  # in ticks
                 max_pitch_jump: int = 12,      # in semitones
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.9):
        
        self.model = model
        self.dataset = dataset
        self.device = device
        self.max_consecutive_rests = max_consecutive_rests
        self.min_note_duration = min_note_duration
        self.max_note_duration = max_note_duration
        self.max_pitch_jump = max_pitch_jump
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Track state during generation
        self.current_pitch = None
        self.rest_count = 0
        self.active_notes = set()
        
    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return logits / self.temperature
    
    def _apply_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        if k > 0:
            values, _ = torch.topk(logits, k)
            min_value = values[-1]
            logits[logits < min_value] = float('-inf')
        return logits
    
    def _apply_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) sampling to logits"""
        if p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
        return logits
    
    def _apply_musical_constraints(self, logits: torch.Tensor, token_history: List[str]) -> torch.Tensor:
        """Apply musical constraints to logits"""
        token_to_id = self.dataset.token_to_id
        id_to_token = self.dataset.id_to_token
        
        # Create mask for all tokens (start with all allowed)
        mask = torch.ones_like(logits, dtype=torch.bool)
        
        last_token = token_history[-1] if token_history else None
        
        # Handle note pitch constraints
        if last_token and last_token.startswith('NOTE_'):
            current_pitch = int(last_token.split('_')[1])
            
            # Limit pitch jumps
            for i in range(len(logits)):
                token = id_to_token[i]
                if token.startswith('NOTE_'):
                    next_pitch = int(token.split('_')[1])
                    if abs(next_pitch - current_pitch) > self.max_pitch_jump:
                        mask[i] = False
        
        # Handle duration constraints
        if last_token and last_token.startswith('NOTE_'):
            # Only allow durations within specified range
            for i in range(len(logits)):
                token = id_to_token[i]
                if token.startswith('DUR_'):
                    duration = int(token.split('_')[1])
                    if duration < self.min_note_duration or duration > self.max_note_duration:
                        mask[i] = False
        
        # Handle rest constraints
        if self.rest_count >= self.max_consecutive_rests:
            # Force a note if too many rests
            for i in range(len(logits)):
                token = id_to_token[i]
                if not token.startswith('NOTE_'):
                    mask[i] = False
        
        # Apply the mask
        logits[~mask] = float('-inf')
        return logits
    
    def generate(self, 
                max_length: int = 512,
                initial_tokens: Optional[List[str]] = None) -> List[str]:
        """Generate a sequence with enhanced constraints and sampling"""
        self.model.eval()
        
        # Initialize with provided tokens or start fresh
        if initial_tokens:
            token_history = initial_tokens
            current_sequence = torch.tensor([
                [self.dataset.token_to_id[t] for t in initial_tokens]
            ], device=self.device)
        else:
            token_history = []
            current_sequence = torch.tensor([[]], device=self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                output = self.model(current_sequence)
                logits = output[0, -1, :]
                
                # Apply temperature and sampling strategies
                logits = self._apply_temperature(logits)
                logits = self._apply_top_k(logits, self.top_k)
                logits = self._apply_top_p(logits, self.top_p)
                
                # Apply musical constraints
                logits = self._apply_musical_constraints(logits, token_history)
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
                
                # Convert to token string
                next_token = self.dataset.id_to_token[next_token_id.item()]
                token_history.append(next_token)
                
                # Update tracking variables
                if next_token.startswith('TIME_'):
                    self.rest_count += 1
                elif next_token.startswith('NOTE_'):
                    self.rest_count = 0
                    self.current_pitch = int(next_token.split('_')[1])
                
                # Add to sequence
                current_sequence = torch.cat([
                    current_sequence, 
                    next_token_id.unsqueeze(0)
                ], dim=1)
        
        return token_history

def generate_music(model, 
                  dataset,
                  output_file: str,
                  temperature: float = 0.8,
                  top_p: float = 0.9,
                  max_length: int = 512,
                  device: str = 'cpu'):
    """High-level function to generate music with enhanced controls"""
    
    generator = EnhancedGenerator(
        model=model,
        dataset=dataset,
        device=device,
        temperature=temperature,
        top_p=top_p,
        max_consecutive_rests=8,
        min_note_duration=120,   # Sixteenth note minimum
        max_note_duration=1920,  # Whole note maximum
        max_pitch_jump=12        # Maximum octave jump
    )
    
    # Generate sequence
    generated_tokens = generator.generate(max_length=max_length)
    
    # Convert to MIDI and save
    sequence_to_midi(generated_tokens, dataset, output_file)
    
    return generated_tokens