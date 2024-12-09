import torch
from torch.utils.data import Dataset
import pretty_midi
from pathlib import Path
import logging
import pickle
import os
from typing import List, Dict, Optional, Tuple


def save_processed_dataset(dataset_obj: object, cache_path: str) -> None:
    """Save processed dataset to a cache file"""
    cache_data = {
        'sequences': dataset_obj.sequences,
        'vocab': dataset_obj.vocab,
        'token_to_id': dataset_obj.token_to_id,
        'id_to_token': dataset_obj.id_to_token,
        'sequence_length': dataset_obj.sequence_length
    }

    # Create directory if it doesn't exist
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    logging.info(f"Saved dataset cache to {cache_path}")


def load_processed_dataset(cache_path: str, sequence_length: int) -> Optional[dict]:
    """Load dataset from cache if valid"""
    try:
        if not os.path.exists(cache_path):
            return None

        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        # Verify cache data is valid
        required_keys = ['sequences', 'vocab', 'token_to_id', 'id_to_token', 'sequence_length']
        if not all(key in cache_data for key in required_keys):
            return None

        # Verify sequence length matches
        if cache_data['sequence_length'] != sequence_length:
            return None

        return cache_data

    except Exception as e:
        logging.warning(f"Failed to load cache: {e}")
        return None


def process_midi_file(midi_path: Path) -> List[str]:
    """Process a single MIDI file into tokens"""
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    tokens = []
    midi_vocab = set()

    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # Skip drum tracks
            notes = sorted(instrument.notes, key=lambda x: x.start)
            for note in notes:
                # Quantize timing to 16th notes
                start_tick = int(note.start * 4)  # Assuming quarter note = 1.0
                duration_tick = max(1, int((note.end - note.start) * 4))
                velocity = note.velocity

                # Add tokens
                time_token = f"TIME_{start_tick}"
                note_token = f"NOTE_{note.pitch}"
                dur_token = f"DUR_{duration_tick}"
                vel_token = f"VEL_{velocity}"

                new_tokens = [time_token, note_token, dur_token, vel_token]
                tokens.extend(new_tokens)
                midi_vocab.update(new_tokens)

    if not tokens:
        raise ValueError(f"No tokens generated from {midi_path}")

    return tokens, midi_vocab


def create_sequences(tokens: List[str], sequence_length: int) -> List[List[str]]:
    """Create overlapping sequences from tokens"""
    sequences = []
    for i in range(0, len(tokens) - sequence_length, sequence_length // 2):
        sequence = tokens[i:i + sequence_length]
        if len(sequence) == sequence_length:
            sequences.append(sequence)
    return sequences


def get_midi_files(midi_folder: str) -> List[Path]:
    """Get all MIDI files in directory recursively"""
    midi_folder = Path(midi_folder)
    if not midi_folder.exists():
        raise FileNotFoundError(f"MIDI folder not found: {midi_folder}")

    midi_files = list(midi_folder.glob('**/*.midi')) + list(midi_folder.glob('**/*.mid'))
    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_folder}")

    return midi_files


def create_vocabulary(vocab: set) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create token-to-id and id-to-token mappings"""
    token_to_id = {token: i for i, token in enumerate(sorted(vocab))}
    id_to_token = {i: token for token, i in token_to_id.items()}
    return token_to_id, id_to_token


def get_cache_path(base_dir: str, sequence_length: int, prefix: str = "dataset") -> str:
    """Generate cache path based on parameters"""
    return os.path.join(base_dir, f"{prefix}_sl{sequence_length}.pkl")