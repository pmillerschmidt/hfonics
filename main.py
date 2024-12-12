import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np

from model import MusicTransformer
from midi_dataset import MIDIDataset
from training_monitor import TrainingMonitor, collect_epoch_metrics, setup_monitoring


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

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint_dir = Path(save_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    # Save regular checkpoint
    path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, path)
    
    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        logging.info(f'Saved best model to {best_path}')
    
    logging.info(f'Saved checkpoint to {path}')

def evaluate(model, val_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            
            total_loss += loss.item()
            total_tokens += y.numel()
    
    # Calculate perplexity and average loss
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity

def train_epoch(model, train_loader, optimizer, device, epoch, monitor, gradient_clip=1.0):
    """Train for one epoch with monitoring"""
    model.train()
    total_loss = 0
    total_tokens = 0
    running_loss = 0
    log_interval = 100

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        batch_tokens = y.numel()
        total_loss += loss.item()
        total_tokens += batch_tokens
        running_loss += loss.item()

        # Calculate metrics for this batch
        batch_loss = loss.item() / batch_tokens
        avg_loss = total_loss / total_tokens
        batch_ppl = np.exp(batch_loss)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': batch_loss,
            'avg_loss': avg_loss,
            'ppl': batch_ppl
        })

        # Log batch metrics
        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            monitor.log_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                loss=batch_loss,
                learning_rate=current_lr
            )
            
            avg_running_loss = running_loss / (log_interval * batch_tokens if batch_idx > 0 else batch_tokens)
            logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_running_loss:.4f}, PPL: {np.exp(avg_running_loss):.2f}')
            running_loss = 0

    epoch_loss = total_loss / total_tokens
    epoch_ppl = np.exp(epoch_loss)
    return epoch_loss, epoch_ppl

# In your monitor class, update the log_batch method:
def log_batch(self, epoch: int, batch_idx: int, loss: float, learning_rate: float):
    """Log batch-level metrics"""
    step = epoch * 1000 + batch_idx  # Global step for tensorboard
    self.writer.add_scalar('Batch/Loss', loss, step)
    self.writer.add_scalar('Batch/Learning_Rate', learning_rate, step)


def main():
    # Training configuration
    config = {
        # Data configuration
        'data_dir': 'maestro-v3.0.0',
        'cache_dir': 'dataset_cache',
        'save_dir': 'runs/melodyforge',
        'validation_split': 0.1,  # 10% for validation
        
        # Musical constraints
        'min_duration_ms': 50.0,
        'max_simultaneous_notes': 6,
        'quantize_duration': True,
        'remove_dissonance': True,
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.0001,
        'sequence_length': 512,
        'gradient_clip': 1.0,
        'warmup_epochs': 5,
        
        # Model configuration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
    }

    # Setup logging
    setup_logging(config['save_dir'])
    logging.info(f'Starting training with config: {config}')

    # Load dataset
    logging.info(f'Loading dataset from {config["data_dir"]}...')
    try:
        full_dataset = MIDIDataset(
            midi_folder=config['data_dir'],
            sequence_length=config['sequence_length'],
            cache_dir=config['cache_dir'],
            min_duration_ms=config['min_duration_ms'],
            max_simultaneous_notes=config['max_simultaneous_notes'],
            quantize_duration=config['quantize_duration'],
            remove_dissonance=config['remove_dissonance']
        )
        logging.info('Splitting dataset into training and validation')
        
        # Split dataset into train and validation
        val_size = int(len(full_dataset) * config['validation_split'])
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logging.info(f'Dataset split: {train_size} training, {val_size} validation sequences')
        logging.info(f'Vocabulary size: {full_dataset.get_vocab_size()}')

    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    # Initialize model
    logging.info('Initializing model...')
    model = MusicTransformer(
        vocab_size=full_dataset.get_vocab_size(),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(config['device'])

    monitor = setup_monitoring(config['save_dir'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # Training loop
    logging.info('Starting training...')
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = 15
    
    monitor = setup_monitoring(config['save_dir'])

    try:
        for epoch in range(config['num_epochs']):
            # Warmup
            if epoch < config['warmup_epochs']:
                warmup_factor = (epoch + 1) / config['warmup_epochs']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['learning_rate'] * warmup_factor
            
            # Train - added monitor argument
            train_loss, train_ppl = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                config['device'], 
                epoch,
                monitor,  # Pass monitor to train_epoch
                config['gradient_clip']
            )
            logging.info(f'Epoch {epoch} Training - Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}')
            
            
            # Validate
            val_loss, val_ppl = evaluate(model, val_loader, config['device'])
            logging.info(f'Epoch {epoch} Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}')

            # Collect metrics for monitoring
            metrics = {
                'train_loss': train_loss,
                'train_perplexity': train_ppl,
                'validation_loss': val_loss,
                'validation_perplexity': val_ppl,
            }

            # Get learning rate before scheduler step
            current_lr = optimizer.param_groups[0]['lr']
            metrics['learning_rate'] = current_lr

            # Learning rate scheduling based on validation loss
            scheduler.step(val_loss)

            # Save checkpoint and check for improvement
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                metrics['best_model'] = True
            else:
                epochs_without_improvement += 1
                metrics['best_model'] = False

            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config['save_dir'], is_best=is_best
            )

            # Log metrics to tensorboard
            monitor.log_epoch(epoch, metrics)
            logging.info(f'Current learning rate: {current_lr}')

            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                logging.info(f'Early stopping triggered after {early_stopping_patience} epochs without improvement')
                break

    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config['save_dir'])
    finally:
        monitor.close()  # Ensure tensorboard writer is closed properly


    logging.info('Training complete!')
    logging.info(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    main()