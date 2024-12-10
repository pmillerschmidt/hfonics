import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import argparse
import pretty_midi

from model import MusicTransformer
from midi_dataset import MIDIDataset

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_dir = Path(save_dir) / 'inference_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'inference_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )

def load_model(checkpoint_path, config, device):
    """Load the trained model from checkpoint"""
    logging.info(f'Loading model from {checkpoint_path}')
    
    # Initialize dataset to get vocabulary size
    dataset = MIDIDataset(
        midi_folder=config['data_dir'],
        sequence_length=config['sequence_length'],
        cache_dir=config['cache_dir']
    )
    
    # Initialize model
    model = MusicTransformer(
        vocab_size=len(dataset.token_to_id),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, dataset

def generate_sequence(model, dataset, seed_sequence=None, max_length=1024, temperature=1.0, device='cuda'):
    """Generate a new music sequence"""
    model.eval()
    
    # Initialize with seed sequence or a random token from the vocabulary
    if seed_sequence is None:
        # Pick a random token from vocabulary as starting point
        initial_token_id = torch.randint(0, len(dataset.token_to_id), (1,))
        current_sequence = torch.tensor([[initial_token_id]], device=device)
    else:
        current_sequence = seed_sequence.unsqueeze(0).to(device)
    
    # Generate tokens one at a time
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            output = model(current_sequence)
            last_token_logits = output[0, -1, :] / temperature
            
            # Sample from the distribution
            probs = torch.softmax(last_token_logits, dim=0)
            next_token = torch.multinomial(probs, 1)
            
            # Add the new token to our sequence
            current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=1)
            
            # Optional: Add some basic heuristics to avoid too long sequences
            if current_sequence.size(1) >= max_length:
                break
    
    return current_sequence.squeeze(0)

def sequence_to_midi(sequence, dataset, output_path):
    """Convert generated token sequence to MIDI file"""
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    current_time = 0.0
    for token in sequence:
        token_str = dataset.id_to_token[token.item()]
        
        # Parse token and add to MIDI
        if token_str.startswith('NOTE_ON'):
            note_num = int(token_str.split('_')[2])
            note = pretty_midi.Note(
                velocity=64,
                pitch=note_num,
                start=current_time,
                end=current_time + 0.5  # Default note duration
            )
            piano.notes.append(note)
        elif token_str.startswith('TIME_SHIFT'):
            time_shift = float(token_str.split('_')[2]) / 1000.0  # Convert to seconds
            current_time += time_shift
    
    pm.instruments.append(piano)
    # Convert Path to string when writing the file
    pm.write(str(output_path))

def main():
    parser = argparse.ArgumentParser(description='Generate music using trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='generated', help='Output directory for generated MIDI files')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    args = parser.parse_args()

    # Configuration (should match training config)
    config = {
        'data_dir': 'maestro-v3.0.0',
        'cache_dir': 'dataset_cache',
        'sequence_length': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
    }

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)
    
    try:
        # Load model
        model, dataset = load_model(args.checkpoint, config, config['device'])
        
        # Generate multiple samples
        for i in range(args.num_samples):
            logging.info(f'Generating sample {i+1}/{args.num_samples}')
            
            # Generate sequence
            generated_sequence = generate_sequence(
                model,
                dataset,
                max_length=args.max_length,
                temperature=args.temperature,
                device=config['device']
            )
            
            # Convert to MIDI and save
            output_path = output_dir / f'generated_{i+1}.mid'
            sequence_to_midi(generated_sequence, dataset, output_path)
            logging.info(f'Saved generated MIDI to {output_path}')
            
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        raise

if __name__ == '__main__':
    main()