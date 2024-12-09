import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

from model import MusicTransformer
from midi_dataset import MIDIDataset


def setup_logging(save_dir):
    """Setup logging configuration"""
    log_dir = Path(save_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    """Save model checkpoint"""
    checkpoint_dir = Path(save_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, path)
    logging.info(f'Saved checkpoint to {path}')


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

        if batch_idx % 100 == 0:
            logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    return total_loss / len(train_loader)


def main():
    # Training configuration
    config = {
        'data_dir': 'maestro-v3.0.0',
        'cache_dir': 'dataset_cache',
        'save_dir': 'runs/melodyforge',
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.0001,
        'sequence_length': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
    }

    # Setup logging
    setup_logging(config['save_dir'])
    logging.info(f'Starting training with config: {config}')

    # Load dataset with more detailed logging
    logging.info(f'Loading dataset from {config["data_dir"]}...')
    try:
        dataset = MIDIDataset(
            midi_folder=config['data_dir'],
            sequence_length=config['sequence_length'],
            cache_dir=config['cache_dir']
        )
        logging.info(f'Successfully loaded dataset with {len(dataset)} sequences')
        logging.info(f'Vocabulary size: {dataset.get_vocab_size()}')

        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Initialize model
    logging.info('Initializing model...')
    model = MusicTransformer(
        vocab_size=len(dataset.token_to_id),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    logging.info('Starting training...')
    best_loss = float('inf')
    try:
        for epoch in range(config['num_epochs']):
            avg_loss = train_epoch(
                model, train_loader, optimizer, config['device'], epoch
            )
            logging.info(f'Epoch {epoch} complete, Average Loss: {avg_loss:.4f}')

            # Learning rate scheduling
            scheduler.step(avg_loss)

            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, epoch, avg_loss, config['save_dir'])

    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
        save_checkpoint(model, optimizer, epoch, avg_loss, config['save_dir'])

    logging.info('Training complete!')


if __name__ == '__main__':
    main()