import torch
from pathlib import Path
import logging
from datetime import datetime
import argparse

from model import MusicTransformer
from midi_dataset import MIDIDataset
from enhanced_generator import EnhancedGenerator, generate_music

"""
python inference.py \
    --checkpoint runs/melodyforge/checkpoints/checkpoint_epoch_5.pt \
    --output generated.mid \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_length 512

"""
def main():
    parser = argparse.ArgumentParser(description='Generate music using trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated.mid', help='Output MIDI file')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    args = parser.parse_args()
    # Configuration (matching your training setup)
    config = {
        'data_dir': 'maestro-v3.0.0',
        'cache_dir': 'dataset_cache',
        'sequence_length': 512,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
    }
    # Load dataset
    dataset = MIDIDataset(
        midi_folder=config['data_dir'],
        sequence_length=config['sequence_length'],
        cache_dir=config['cache_dir']
    )
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MusicTransformer(
        vocab_size=dataset.get_vocab_size(),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Generate music
    logging.info("Generating music...")
    generated_tokens = generate_music(
        model=model,
        dataset=dataset,
        output_file=args.output,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length,
        device=device
    )
    logging.info(f"Generated music saved to: {args.output}")
    # Print some statistics about the generated sequence
    note_count = len([t for t in generated_tokens if t.startswith('NOTE_')])
    time_shifts = len([t for t in generated_tokens if t.startswith('TIME_')])
    logging.info(f"Generation statistics:")
    logging.info(f"Total tokens: {len(generated_tokens)}")
    logging.info(f"Note events: {note_count}")
    logging.info(f"Time shifts: {time_shifts}")

if __name__ == '__main__':
    main()