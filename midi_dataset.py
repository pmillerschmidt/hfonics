import torch
from torch.utils.data import Dataset
import logging
from typing import Optional, List, Tuple, Dict, Set
import os
import pretty_midi
import numpy as np
from pathlib import Path
import hashlib

def get_cache_path(cache_dir: str, sequence_length: int) -> str:
    """Generate a cache path based on sequence length"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    params_str = f"seq_{sequence_length}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return cache_dir / f"dataset_cache_{params_hash}.pt"

def get_midi_files(midi_folder: str) -> List[Path]:
    """Get all MIDI files in the folder and subfolders"""
    midi_folder = Path(midi_folder)
    midi_files = []
    for ext in ['*.mid', '*.midi']:
        midi_files.extend(midi_folder.rglob(ext))
    return midi_files

def create_sequences(tokens: List[str], sequence_length: int) -> List[List[str]]:
    """Create sequences of fixed length from token list"""
    sequences = []
    for i in range(0, len(tokens) - sequence_length, 3):  # Step by 3 to maintain note grouping
        sequence = tokens[i:i + sequence_length + 1]  # +1 for target
        if len(sequence) == sequence_length + 1:
            sequences.append(sequence)
    return sequences

class MIDIDataset(Dataset):
    def __init__(self,
                 midi_folder: str,
                 sequence_length: int = 256,
                 cache_dir: Optional[str] = None,
                 min_duration_ms: float = 50.0,
                 max_simultaneous_notes: int = 6,
                 quantize_duration: bool = True,
                 remove_dissonance: bool = True):
        
        self.sequence_length = sequence_length
        self.midi_folder = midi_folder
        self.cache_path = get_cache_path(cache_dir or "cache", sequence_length) if cache_dir else None
        
        # Musical parameters
        self.ticks_per_beat = 480
        self.min_duration_ticks = int(min_duration_ms * self.ticks_per_beat / 1000)
        self.max_simultaneous_notes = max_simultaneous_notes
        self.should_quantize_duration = quantize_duration
        self.should_remove_dissonance = remove_dissonance
        
        # Musical constants
        self.duration_grid = np.array([120, 240, 360, 480, 720, 960, 1440, 1920])
        self.consonant_intervals = {0, 3, 4, 5, 7, 8, 9, 12}  # Common musical intervals
        
        # Try loading from cache
        if self.cache_path:
            cache_data = self._load_from_cache(self.cache_path)
            if cache_data:
                self.sequences = cache_data['sequences']
                self.vocab = cache_data['vocab']
                self.token_to_id = cache_data['token_to_id']
                self.id_to_token = cache_data['id_to_token']
                logging.info(f"Loaded dataset from cache with {len(self.sequences)} sequences")
                return

        # Process data if no cache
        self._process_data()
        
        # Save to cache
        if self.cache_path:
            self._save_to_cache()
    
    def _quantize_duration(self, duration: int) -> int:
        """Quantize a duration to the nearest musical duration"""
        idx = np.argmin(np.abs(self.duration_grid - duration))
        return int(self.duration_grid[idx])
    
    def _is_consonant(self, notes: List[int]) -> bool:
        """Check if a set of notes forms consonant intervals"""
        if len(notes) <= 1:
            return True
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                interval = abs(notes[i] - notes[j]) % 12
                if interval not in self.consonant_intervals:
                    return False
        return True
    
    def process_midi_file(self, midi_path) -> Tuple[List[str], Set[str]]:
        """Process a MIDI file into tokens with musical constraints"""
        try:
            midi_path_str = str(midi_path) if isinstance(midi_path, Path) else midi_path
            midi_data = pretty_midi.PrettyMIDI(midi_path_str)
            tokens = []
            vocab = set()
            
            # Extract and sort notes
            all_notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        start_tick = int(note.start * self.ticks_per_beat * 2)
                        duration_tick = int((note.end - note.start) * self.ticks_per_beat * 2)
                        
                        if duration_tick >= self.min_duration_ticks:
                            all_notes.append({
                                'start': start_tick,
                                'end': start_tick + duration_tick,
                                'pitch': note.pitch,
                                'velocity': note.velocity,
                                'duration': duration_tick
                            })
            
            all_notes.sort(key=lambda x: (x['start'], x['pitch']))
            
            # Process notes with constraints
            current_time = 0
            current_chord = []
            
            for note in all_notes:
                # Add time shift if needed
                if note['start'] > current_time:
                    time_shift = note['start'] - current_time
                    time_token = f"TIME_{time_shift}"
                    tokens.append(time_token)
                    vocab.add(time_token)
                    current_time = note['start']
                    current_chord = []
                
                # Apply musical constraints
                proposed_chord = current_chord + [note['pitch']]
                if (len(proposed_chord) <= self.max_simultaneous_notes and 
                    (not self.should_remove_dissonance or self._is_consonant(proposed_chord))):
                    
                    duration = (self._quantize_duration(note['duration']) 
                              if self.should_quantize_duration 
                              else note['duration'])
                    
                    note_tokens = [
                        f"NOTE_{note['pitch']}",
                        f"DUR_{duration}",
                        f"VEL_{note['velocity']}"
                    ]
                    tokens.extend(note_tokens)
                    vocab.update(note_tokens)
                    current_chord.append(note['pitch'])
            
            if not tokens:
                logging.warning(f"No tokens generated from {midi_path}")
                
            return tokens, vocab
            
        except Exception as e:
            logging.error(f"Error processing {midi_path}: {e}")
            return [], set()
    
    def _process_data(self) -> None:
        """Process all MIDI files"""
        logging.info("Processing MIDI files...")
        self.sequences = []
        self.vocab = set()
        
        midi_files = get_midi_files(self.midi_folder)
        logging.info(f"Found {len(midi_files)} MIDI files")
        
        for midi_file in midi_files:
            try:
                tokens, file_vocab = self.process_midi_file(midi_file)
                self.vocab.update(file_vocab)
                
                sequences = create_sequences(tokens, self.sequence_length)
                self.sequences.extend(sequences)
                
                logging.info(f"Processed {midi_file}: {len(sequences)} sequences")
            except Exception as e:
                logging.error(f"Error processing {midi_file}: {e}")
                continue
        
        if not self.sequences:
            raise ValueError("No valid sequences were created from the MIDI files")
        
        # Create vocabulary mappings
        sorted_vocab = sorted(list(self.vocab))
        self.token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        logging.info(f"Created dataset with {len(self.sequences)} sequences and {len(self.vocab)} tokens")
    
    def _load_from_cache(self, cache_path) -> Optional[Dict]:
        """Load dataset from cache if available"""
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path)
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
        return None
    
    def _save_to_cache(self) -> None:
        """Save processed dataset to cache"""
        logging.info(f"Caching dataset in {self.cache_path}")
        cache_data = {
            'sequences': self.sequences,
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token
        }
        torch.save(cache_data, self.cache_path)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple:
        tokens = self.sequences[idx]
        ids = [self.token_to_id[token] for token in tokens]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y
    
    def get_vocab_size(self) -> int:
        return len(self.token_to_id)