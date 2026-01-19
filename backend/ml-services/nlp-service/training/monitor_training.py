#!/usr/bin/env python3
"""
Live training progress monitor
Extracts and displays training progress from log files
"""
import os
import sys
import re
import time
from pathlib import Path

def extract_progress_from_log(log_file: str):
    """Extract training progress from log file"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for progress bars (e.g., "67%|██████▋   | 18/27")
        progress_info = {
            'current_step': None,
            'total_steps': None,
            'percentage': None,
            'epoch': None,
            'loss': None,
            'accuracy': None,
            'last_update': None,
            'status': 'unknown'
        }
        
        # Match progress bar: "67%|██████▋   | 18/27 [01:41<00:46,  5.19s/it]"
        # Try multiple patterns
        progress_patterns = [
            r'(\d+)%\s*\|\s*[█▉▊▋▌▍▎▏\s]+\s*\|\s*(\d+)/(\d+)',  # Standard progress bar
            r'(\d+)%\s*\|\s*(\d+)/(\d+)',  # Simpler pattern
        ]
        
        for pattern in progress_patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                last_match = matches[-1]
                progress_info['percentage'] = int(last_match.group(1))
                progress_info['current_step'] = int(last_match.group(2))
                progress_info['total_steps'] = int(last_match.group(3))
                # Get context around the match
                start = max(0, last_match.start() - 50)
                end = min(len(content), last_match.end() + 50)
                progress_info['last_update'] = content[start:end].strip()
                progress_info['status'] = 'training'
                break
        
        # Match epoch info: "'epoch': 1.0" or "epoch': 1.0"
        epoch_matches = re.findall(r"['\"]?epoch['\"]?:\s*([\d.]+)", content)
        if epoch_matches:
            progress_info['epoch'] = float(epoch_matches[-1])
        
        # Match metrics: "'eval_accuracy': 0.9333"
        acc_matches = re.findall(r"['\"]?eval_accuracy['\"]?:\s*([\d.]+)", content)
        if acc_matches:
            progress_info['accuracy'] = float(acc_matches[-1])
        
        loss_matches = re.findall(r"['\"]?eval_loss['\"]?:\s*([\d.]+)", content)
        if loss_matches:
            progress_info['loss'] = float(loss_matches[-1])
        
        # Check for completion indicators
        if 'Training completed' in content or 'TRAINING SUMMARY' in content:
            progress_info['status'] = 'completed'
        elif 'Starting training' in content and progress_info['percentage'] is None:
            progress_info['status'] = 'starting'
        
        return progress_info
    except Exception as e:
        return {'error': str(e), 'status': 'error'}

def check_checkpoint_progress(model_dir: str):
    """Check progress based on checkpoint directories"""
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(model_dir):
        if item.startswith('checkpoint-'):
            try:
                step_num = int(item.replace('checkpoint-', ''))
                checkpoints.append(step_num)
            except:
                pass
    
    if checkpoints:
        latest_checkpoint = max(checkpoints)
        return {
            'latest_checkpoint': latest_checkpoint,
            'checkpoint_count': len(checkpoints)
        }
    return None

def monitor_training():
    """Monitor training progress"""
    base_dir = Path(__file__).parent.parent
    
    phishing_log = base_dir / '..' / '..' / 'tmp' / 'phishing_train_final.log'
    ai_log = base_dir / '..' / '..' / 'tmp' / 'ai_train_final.log'
    
    phishing_log = phishing_log.resolve()
    ai_log = ai_log.resolve()
    
    print("=" * 80)
    print("TRAINING PROGRESS MONITOR")
    print("=" * 80)
    print()
    
    # Phishing Model Status
    print("📧 PHISHING MODEL TRAINING")
    print("-" * 80)
    phishing_progress = extract_progress_from_log(str(phishing_log))
    phishing_checkpoint = check_checkpoint_progress(str(base_dir.parent / 'models' / 'phishing-bert-v1'))
    
    if phishing_progress and 'error' not in phishing_progress:
        if phishing_progress['percentage'] is not None:
            print(f"  Progress: {phishing_progress['percentage']}%")
            if phishing_progress['current_step'] and phishing_progress['total_steps']:
                print(f"  Steps: {phishing_progress['current_step']}/{phishing_progress['total_steps']}")
            if phishing_progress['epoch']:
                print(f"  Epoch: {phishing_progress['epoch']:.1f}/3.0")
            if phishing_progress['accuracy']:
                print(f"  Validation Accuracy: {phishing_progress['accuracy']:.4f}")
            if phishing_progress['loss']:
                print(f"  Validation Loss: {phishing_progress['loss']:.4f}")
        else:
            print("  Status: Training completed or not started")
    elif phishing_checkpoint:
        print(f"  Status: Checkpoints found (latest: checkpoint-{phishing_checkpoint['latest_checkpoint']})")
        print("  Model files exist - Training likely completed")
    else:
        print("  Status: No active training detected")
    
    # Check if model files exist
    model_path = base_dir.parent / 'models' / 'phishing-bert-v1' / 'model.safetensors'
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model saved: {size_mb:.1f} MB")
    
    print()
    
    # AI Detector Status
    print("🤖 AI DETECTOR TRAINING")
    print("-" * 80)
    ai_progress = extract_progress_from_log(str(ai_log))
    ai_checkpoint = check_checkpoint_progress(str(base_dir.parent / 'models' / 'ai-detector-v1'))
    
    if ai_progress and 'error' not in ai_progress:
        if ai_progress['percentage'] is not None:
            print(f"  Progress: {ai_progress['percentage']}%")
            if ai_progress['current_step'] and ai_progress['total_steps']:
                print(f"  Steps: {ai_progress['current_step']}/{ai_progress['total_steps']}")
                remaining = ai_progress['total_steps'] - ai_progress['current_step']
                print(f"  Remaining: {remaining} steps")
            if ai_progress['epoch']:
                print(f"  Epoch: {ai_progress['epoch']:.1f}/3.0")
            if ai_progress['accuracy']:
                print(f"  Validation Accuracy: {ai_progress['accuracy']:.4f}")
            if ai_progress['loss']:
                print(f"  Validation Loss: {ai_progress['loss']:.4f}")
        else:
            print("  Status: Training in progress (checking logs...)")
            if ai_progress.get('last_update'):
                print(f"  Last update: {ai_progress['last_update'][:100]}...")
    elif ai_checkpoint:
        print(f"  Status: Checkpoints found (latest: checkpoint-{ai_checkpoint['latest_checkpoint']})")
    else:
        print("  Status: Training may have stopped or not started")
    
    # Check if model files exist
    ai_model_path = base_dir.parent / 'models' / 'ai-detector-v1' / 'model.safetensors'
    if ai_model_path.exists():
        size_mb = ai_model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model saved: {size_mb:.1f} MB")
    else:
        print("  ⏳ Model not yet saved")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    monitor_training()
