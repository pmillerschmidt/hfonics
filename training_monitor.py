import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict

class TrainingMonitor:
    def __init__(self, log_dir: str, model_name: str = "music_transformer"):
        """Initialize training monitor with TensorBoard writer"""
        self.log_dir = Path(log_dir) / model_name / "tensorboard"
        self.writer = SummaryWriter(self.log_dir)
        self.metrics = defaultdict(list)
        self.current_epoch = 0
        
    def log_batch(self, epoch: int, batch_idx: int, loss: float, learning_rate: float):
        """Log batch-level metrics"""
        step = epoch * 1000 + batch_idx  # Global step for tensorboard
        self.writer.add_scalar('Batch/Loss', loss, step)
        self.writer.add_scalar('Batch/Learning_Rate', learning_rate, step)
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch-level metrics and generate visualizations"""
        self.current_epoch = epoch
        
        # Log basic metrics
        for name, value in metrics.items():
            self.metrics[name].append(value)
            self.writer.add_scalar(f'Epoch/{name}', value, epoch)
        
        # Log learning curves
        self._plot_learning_curves()
        
        # Log token distribution if available
        if 'token_dist' in metrics:
            self._plot_token_distribution(metrics['token_dist'], epoch)
        
        # Log attention weights if available
        if 'attention_weights' in metrics:
            self._plot_attention_heatmap(metrics['attention_weights'], epoch)
    
    def _plot_learning_curves(self):
        """Plot and save learning curves"""
        plt.figure(figsize=(12, 6))
        for name, values in self.metrics.items():
            if 'loss' in name.lower() or 'perplexity' in name.lower():
                plt.plot(values, label=name)
        
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        # Save to TensorBoard
        self.writer.add_figure('Learning_Curves', plt.gcf(), self.current_epoch)
        plt.close()
    
    def _plot_token_distribution(self, token_dist, epoch):
        """Plot token distribution as histogram"""
        plt.figure(figsize=(15, 5))
        sns.histplot(data=token_dist, bins=50)
        plt.title('Token Distribution')
        plt.xlabel('Token Type')
        plt.ylabel('Frequency')
        
        self.writer.add_figure('Token_Distribution', plt.gcf(), epoch)
        plt.close()
    
    def _plot_attention_heatmap(self, attention_weights, epoch):
        """Plot attention weights as heatmap"""
        plt.figure(figsize=(10, 10))
        sns.heatmap(attention_weights, cmap='viridis')
        plt.title('Attention Weights')
        plt.xlabel('Query Position')
        plt.ylabel('Key Position')
        
        self.writer.add_figure('Attention_Weights', plt.gcf(), epoch)
        plt.close()
    
    def log_model_prediction(self, input_seq, output_seq, epoch):
        """Log model prediction examples"""
        # Convert token sequences to readable format
        prediction_text = f"Input: {input_seq}\nOutput: {output_seq}"
        self.writer.add_text('Predictions', prediction_text, epoch)
    
    def compute_musical_metrics(self, generated_sequence):
        """Compute music-specific metrics"""
        metrics = {}
        
        # Note density (notes per time unit)
        note_events = [t for t in generated_sequence if t.startswith('NOTE_')]
        metrics['note_density'] = len(note_events) / len(generated_sequence)
        
        # Average velocity
        vel_events = [int(t.split('_')[1]) for t in generated_sequence if t.startswith('VEL_')]
        metrics['avg_velocity'] = np.mean(vel_events) if vel_events else 0
        
        # Duration distribution
        dur_events = [int(t.split('_')[1]) for t in generated_sequence if t.startswith('DUR_')]
        metrics['avg_duration'] = np.mean(dur_events) if dur_events else 0
        
        return metrics
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

def collect_epoch_metrics(model, epoch_loss, epoch_ppl, val_loss, val_ppl, 
                         generated_sequence=None, attention_weights=None):
    """Collect all metrics for one epoch"""
    metrics = {
        'train_loss': epoch_loss,
        'train_perplexity': epoch_ppl,
        'validation_loss': val_loss,
        'validation_perplexity': val_ppl,
    }
    
    # Add attention visualization if available
    if attention_weights is not None:
        metrics['attention_weights'] = attention_weights
    
    # Add token distribution if sequence is available
    if generated_sequence is not None:
        token_counts = defaultdict(int)
        for token in generated_sequence:
            token_type = token.split('_')[0]
            token_counts[token_type] += 1
        metrics['token_dist'] = token_counts
    
    return metrics

def setup_monitoring(save_dir: str):
    """Setup monitoring for training"""
    monitor = TrainingMonitor(save_dir)
    return monitor