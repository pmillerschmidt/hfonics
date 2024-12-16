import torch
import pretty_midi
from pathlib import Path
import logging
from datetime import datetime

from model import MusicTransformer
from midi_dataset import MIDIDataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_model(checkpoint_path, config, device='cpu'):
    logging.info(f'Loading model from {checkpoint_path}')
    # Only load vocabulary info if possible
    dataset = MIDIDataset(
        midi_folder=config['data_dir'],
        sequence_length=config['sequence_length'],
        cache_dir=config['cache_dir']
    )
    model = MusicTransformer(
        vocab_size=len(dataset.token_to_id),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, dataset

def print_midi_events(sequence, dataset):
    print("\nSequence Analysis:")
    print("-" * 50)
    current_time = 0.0
    current_note = None
    current_velocity = None
    current_duration = None
    events = []
    # Process tokens into note events
    for i, token_id in enumerate(sequence):
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
            # If we have all components of a note, add it to events
            if all(x is not None for x in [current_note, current_velocity, current_duration]):
                events.append({
                    'time': current_time,
                    'note': current_note,
                    'velocity': current_velocity,
                    'duration': current_duration,
                    'note_name': pretty_midi.note_number_to_name(current_note)
                })
                # Reset note components
                current_note = None
                current_velocity = None
                current_duration = None
    # Print sequence statistics
    print(f"Total sequence length: {len(sequence)} tokens")
    print(f"Total duration: {current_time:.1f} ticks")
    print(f"Number of complete notes: {len(events)}")
    # Print the first 20 note events
    print("\nFirst 20 Note Events:")
    print("-" * 50)
    print("Time(ticks)  Note   Pitch  Vel  Duration")
    print("-" * 50)
    for i, event in enumerate(events[:20]):
        print(f"{event['time']:10.1f}  {event['note_name']:>4}  {event['note']:4d}  {event['velocity']:3d}  {event['duration']:8.1f}")


def convert_sequence_to_midi(sequence, dataset, output_path):
    midi_data = pretty_midi.PrettyMIDI()
    # Create instrument instance (Piano by default)
    instrument = pretty_midi.Instrument(program=0)  # 0 is Acoustic Grand Piano
    # Process tokens into note events
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
            # If we have all components of a note, create a MIDI note
            if all(x is not None for x in [current_note, current_velocity, current_duration]):
                note = pretty_midi.Note(
                    velocity=current_velocity,
                    pitch=current_note,
                    start=current_time / 1000,
                    end=(current_time + current_duration) / 1000
                )
                instrument.notes.append(note)
                current_note = None
                current_velocity = None
                current_duration = None
    midi_data.instruments.append(instrument)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(output_path)
    logging.info(f"MIDI file saved to {output_path}")


def generate_and_analyze():
    config = {
        'data_dir': 'maestro-v3.0.0',
        'cache_dir': 'dataset_cache',
        'sequence_length': 512,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
    }
    checkpoint_path = "checkpoints/checkpoint_epoch_9.pt"
    setup_logging()
    try:
        # Load model
        model, dataset = load_model(checkpoint_path, config, device='cpu')
        # Generate sequence
        logging.info("Generating music sequence...")
        with torch.no_grad():
            initial_token_id = torch.randint(0, len(dataset.token_to_id), (1,))
            current_sequence = torch.tensor([[initial_token_id]])
            max_length = 512
            temperature = 1.0
            for _ in range(max_length):
                output = model(current_sequence)
                last_token_logits = output[0, -1, :] / temperature
                probs = torch.softmax(last_token_logits, dim=0)
                next_token = torch.multinomial(probs, 1)
                current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=1)
        generated_sequence = current_sequence.squeeze(0)
        print("\nRaw Token Sequence (first 40 tokens):")
        print("-" * 50)
        for i, token_id in enumerate(generated_sequence[:40]):
            token_str = dataset.id_to_token[token_id.item()]
            print(f"[{i:3d}] {token_str}")
        print_midi_events(generated_sequence, dataset)
        output_filename = f"generated_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
        output_path = Path("generated_midi") / output_filename
        convert_sequence_to_midi(generated_sequence, dataset, str(output_path))
    except Exception as e:
        logging.error(f"Error during generation/analysis: {e}")
        raise

if __name__ == '__main__':
    generate_and_analyze()