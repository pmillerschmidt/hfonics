import torch
from torch.utils.data import Dataset
import logging
from typing import Optional

from dataset_utils import (
    save_processed_dataset,
    load_processed_dataset,
    process_midi_file,
    create_sequences,
    get_midi_files,
    create_vocabulary,
    get_cache_path
)


class MIDIDataset(Dataset):
    def __init__(self,
                 midi_folder: str,
                 sequence_length: int = 256,
                 cache_dir: Optional[str] = None):
        self.sequence_length = sequence_length
        self.midi_folder = midi_folder
        self.cache_path = get_cache_path(cache_dir or "cache", sequence_length) if cache_dir else None

        # Try loading from cache
        if self.cache_path:
            cache_data = load_processed_dataset(self.cache_path, sequence_length)
            if cache_data:
                self._load_from_cache(cache_data)
                logging.info(f"Loaded dataset from cache with {len(self.sequences)} sequences")
                return

        # Process MIDI files if no cache or cache invalid
        self._process_data()

        # Save to cache if path provided
        if self.cache_path:
            save_processed_dataset(self, self.cache_path)

    def _load_from_cache(self, cache_data: dict) -> None:
        """Load dataset attributes from cache data"""
        self.sequences = cache_data['sequences']
        self.vocab = cache_data['vocab']
        self.token_to_id = cache_data['token_to_id']
        self.id_to_token = cache_data['id_to_token']

    def _process_data(self) -> None:
        """Process all MIDI files from scratch"""
        logging.info("Processing MIDI files...")
        self.sequences = []
        self.vocab = set()

        midi_files = get_midi_files(self.midi_folder)
        logging.info(f"Found {len(midi_files)} MIDI files")

        for midi_file in midi_files:
            try:
                tokens, file_vocab = process_midi_file(midi_file)
                self.vocab.update(file_vocab)

                sequences = create_sequences(tokens, self.sequence_length)
                self.sequences.extend(sequences)

                logging.info(f"Processed {midi_file}: {len(sequences)} sequences")

            except Exception as e:
                logging.error(f"Error processing {midi_file}: {e}")
                continue

        if not self.sequences:
            raise ValueError("No valid sequences were created from the MIDI files")

        self.token_to_id, self.id_to_token = create_vocabulary(self.vocab)
        logging.info(f"Created dataset with {len(self.sequences)} sequences and {len(self.vocab)} tokens")

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