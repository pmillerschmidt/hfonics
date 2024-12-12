import os
from pathlib import Path
import hashlib
import torch
import pretty_midi
import logging
from typing import List, Set, Tuple, Dict
from glob import glob

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

def process_midi_file(midi_path: str) -> Tuple[List[str], Set[str]]:
    """Process a MIDI file into a sequence of tokens"""
    try:
        # Convert Path to string for pretty_midi
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        tokens = []
        vocab = set()
        
        # Extract notes and sort by start time
        all_notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Skip drum tracks
                for note in instrument.notes:
                    # Convert time to ticks (assume 480 ticks per beat)
                    start_tick = int(note.start * 480)
                    duration_tick = int((note.end - note.start) * 480)
                    all_notes.append({
                        'start': start_tick,
                        'pitch': note.pitch,
                        'duration': duration_tick,
                        'velocity': note.velocity
                    })
        
        # Sort notes by start time
        all_notes.sort(key=lambda x: x['start'])
        
        # Convert to tokens
        current_time = 0
        for note in all_notes:
            # Add time shift if needed
            if note['start'] > current_time:
                time_shift = note['start'] - current_time
                time_token = f"TIME_{time_shift}"
                tokens.append(time_token)
                vocab.add(time_token)
                current_time = note['start']
            
            # Add note tokens
            note_tokens = [
                f"NOTE_{note['pitch']}",
                f"DUR_{note['duration']}",
                f"VEL_{note['velocity']}"
            ]
            tokens.extend(note_tokens)
            vocab.update(note_tokens)
        
        return tokens, vocab
        
    except Exception as e:
        logging.error(f"Error processing {midi_path}: {e}")
        return [], set()

def create_sequences(tokens: List[str], sequence_length: int) -> List[List[str]]:
    """Create sequences of fixed length from token list"""
    sequences = []
    for i in range(0, len(tokens) - sequence_length, 3):  # Step by 3 to maintain note grouping
        sequence = tokens[i:i + sequence_length + 1]  # +1 for target
        if len(sequence) == sequence_length + 1:
            sequences.append(sequence)
    return sequences

def create_vocabulary(vocab: Set[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create token-to-id and id-to-token mappings"""
    sorted_vocab = sorted(list(vocab))
    token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}
    return token_to_id, id_to_token

def save_processed_dataset(dataset, cache_path: str) -> None:
    """Save processed dataset to cache"""
    cache_data = {
        'sequences': dataset.sequences,
        'vocab': dataset.vocab,
        'token_to_id': dataset.token_to_id,
        'id_to_token': dataset.id_to_token
    }
    torch.save(cache_data, cache_path)

def load_processed_dataset(cache_path: str, sequence_length: int) -> Dict:
    """Load processed dataset from cache if valid"""
    if os.path.exists(cache_path):
        try:
            return torch.load(cache_path)
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            return None
    return None