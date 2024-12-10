import torch
import pretty_midi
from pathlib import Path
import logging

from model import MusicTransformer
from midi_dataset import MIDIDataset

def sequence_to_midi(sequence, dataset, output_path, ticks_per_beat=480):
    """Convert token sequence to MIDI file with proper timing"""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)  # Standard tempo
    piano = pretty_midi.Instrument(program=0)  # Grand Piano
    
    current_time = 0.0
    current_note = None
    current_velocity = None
    current_duration = None
    
    for token_id in sequence:
        token_str = dataset.id_to_token[token_id.item()]
        
        if token_str.startswith('TIME_'):
            time_delta = float(token_str.split('_')[1])
            current_time += time_delta
        elif token_str.startswith('NOTE_'):
            current_note = int(token_str.split('_')[1])
        elif token_str.startswith('VEL_'):
            current_velocity = int(token_str.split('_')[1])
        elif token_str.startswith('DUR_'):
            current_duration = float(token_str.split('_')[1])
            
            # If we have all components of a note, create it
            if all(x is not None for x in [current_note, current_velocity, current_duration]):
                # Convert ticks to seconds (assuming default tempo)
                start_time = current_time / ticks_per_beat * 0.5  # 0.5 seconds per beat
                end_time = (current_time + current_duration) / ticks_per_beat * 0.5
                
                note = pretty_midi.Note(
                    velocity=current_velocity,
                    pitch=current_note,
                    start=start_time,
                    end=end_time
                )
                piano.notes.append(note)
                
                # Reset note components
                current_note = None
                current_velocity = None
                current_duration = None
    
    pm.instruments.append(piano)
    pm.write(str(output_path))
    return pm

def generate_midi(model, dataset, output_path, max_length=512, temperature=1.0, device='cpu'):
    """Generate a sequence and convert it to MIDI"""
    model.eval()
    
    with torch.no_grad():
        initial_token_id = torch.randint(0, len(dataset.token_to_id), (1,))
        current_sequence = torch.tensor([[initial_token_id]], device=device)
        
        for _ in range(max_length):
            output = model(current_sequence)
            last_token_logits = output[0, -1, :] / temperature
            probs = torch.softmax(last_token_logits, dim=0)
            next_token = torch.multinomial(probs, 1)
            current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=1)
    
    generated_sequence = current_sequence.squeeze(0)
    midi_file = sequence_to_midi(generated_sequence, dataset, output_path)
    return generated_sequence, midi_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MIDI from trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated.mid', help='Output MIDI file')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--length', type=int, default=512, help='Sequence length')
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
    
    # Load dataset first
    dataset = MIDIDataset(
        midi_folder=config['data_dir'],
        sequence_length=config['sequence_length'],
        cache_dir=config['cache_dir']
    )
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MusicTransformer(
        vocab_size=len(dataset.token_to_id),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate MIDI
    sequence, midi_file = generate_midi(
        model,
        dataset,
        args.output,
        max_length=args.length,
        temperature=args.temperature,
        device=device
    )
    print(f"Generated MIDI file saved to: {args.output}")

if __name__ == '__main__':
    main()