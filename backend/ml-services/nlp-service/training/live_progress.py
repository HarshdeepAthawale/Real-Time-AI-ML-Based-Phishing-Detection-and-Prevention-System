#!/usr/bin/env python3
"""
Live training progress monitor with continuous updates
"""
import os
import sys
import re
import time
from pathlib import Path
from datetime import datetime

def extract_progress_from_log(log_file: str):
    """Extract training progress from log file"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        progress_info = {
            'current_step': None,
            'total_steps': None,
            'percentage': None,
            'epoch': None,
            'loss': None,
            'accuracy': None,
            'status': 'unknown'
        }
        
        # Match progress bar patterns
        progress_patterns = [
            r'(\d+)%\s*\|\s*[█▉▊▋▌▍▎▏\s]+\s*\|\s*(\d+)/(\d+)',
            r'(\d+)%\s*\|\s*(\d+)/(\d+)',
        ]
        
        for pattern in progress_patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                last_match = matches[-1]
                progress_info['percentage'] = int(last_match.group(1))
                progress_info['current_step'] = int(last_match.group(2))
                progress_info['total_steps'] = int(last_match.group(3))
                progress_info['status'] = 'training'
                break
        
        # Extract epoch
        epoch_matches = re.findall(r"['\"]?epoch['\"]?:\s*([\d.]+)", content)
        if epoch_matches:
            progress_info['epoch'] = float(epoch_matches[-1])
        
        # Extract metrics
        acc_matches = re.findall(r"['\"]?eval_accuracy['\"]?:\s*([\d.]+)", content)
        if acc_matches:
            progress_info['accuracy'] = float(acc_matches[-1])
        
        loss_matches = re.findall(r"['\"]?eval_loss['\"]?:\s*([\d.]+)", content)
        if loss_matches:
            progress_info['loss'] = float(loss_matches[-1])
        
        # Check status
        if 'Training completed' in content or 'TRAINING SUMMARY' in content:
            progress_info['status'] = 'completed'
        elif 'Starting training' in content and progress_info['percentage'] is None:
            progress_info['status'] = 'starting'
        
        return progress_info
    except Exception as e:
        return {'error': str(e), 'status': 'error'}

def format_progress_bar(percentage, width=40):
    """Create a visual progress bar"""
    if percentage is None:
        return "[" + " " * width + "]"
    filled = int(width * percentage / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage}%"

def monitor_live():
    """Live monitoring with continuous updates"""
    base_dir = Path(__file__).parent.parent
    
    phishing_log = base_dir.parent / 'tmp' / 'phishing_train_final.log'
    ai_log = base_dir.parent / 'tmp' / 'ai_train_final.log'
    
    phishing_model = base_dir.parent / 'models' / 'phishing-bert-v1' / 'model.safetensors'
    ai_model = base_dir.parent / 'models' / 'ai-detector-v1' / 'model.safetensors'
    
    print("\033[2J\033[H")  # Clear screen
    print("=" * 80)
    print("LIVE TRAINING PROGRESS MONITOR")
    print("=" * 80)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            # Phishing Model
            print("\033[1m📧 PHISHING MODEL\033[0m")
            print("-" * 80)
            
            if phishing_model.exists():
                size_mb = phishing_model.stat().st_size / (1024 * 1024)
                print(f"  ✓ Training COMPLETED")
                print(f"  Model size: {size_mb:.1f} MB")
                print(f"  Location: models/phishing-bert-v1/")
            else:
                phishing_progress = extract_progress_from_log(str(phishing_log))
                if phishing_progress and 'error' not in phishing_progress:
                    if phishing_progress['percentage'] is not None:
                        print(f"  {format_progress_bar(phishing_progress['percentage'])}")
                        if phishing_progress['current_step'] and phishing_progress['total_steps']:
                            remaining = phishing_progress['total_steps'] - phishing_progress['current_step']
                            print(f"  Steps: {phishing_progress['current_step']}/{phishing_progress['total_steps']} (Remaining: {remaining})")
                        if phishing_progress['epoch']:
                            print(f"  Epoch: {phishing_progress['epoch']:.1f}/3.0")
                        if phishing_progress['accuracy']:
                            print(f"  Validation Accuracy: {phishing_progress['accuracy']:.4f}")
                        if phishing_progress['loss']:
                            print(f"  Validation Loss: {phishing_progress['loss']:.4f}")
                    else:
                        print(f"  Status: {phishing_progress.get('status', 'unknown')}")
                else:
                    print("  Status: No training detected")
            print()
            
            # AI Detector
            print("\033[1m🤖 AI DETECTOR\033[0m")
            print("-" * 80)
            
            if ai_model.exists():
                size_mb = ai_model.stat().st_size / (1024 * 1024)
                print(f"  ✓ Training COMPLETED")
                print(f"  Model size: {size_mb:.1f} MB")
                print(f"  Location: models/ai-detector-v1/")
            else:
                ai_progress = extract_progress_from_log(str(ai_log))
                if ai_progress and 'error' not in ai_progress:
                    if ai_progress['percentage'] is not None:
                        print(f"  {format_progress_bar(ai_progress['percentage'])}")
                        if ai_progress['current_step'] and ai_progress['total_steps']:
                            remaining = ai_progress['total_steps'] - ai_progress['current_step']
                            print(f"  Steps: {ai_progress['current_step']}/{ai_progress['total_steps']} (Remaining: {remaining})")
                            # Estimate time remaining (rough estimate: 0.5s per step)
                            if remaining > 0:
                                est_seconds = remaining * 0.5
                                est_minutes = est_seconds / 60
                                print(f"  Estimated time remaining: ~{est_minutes:.1f} minutes")
                        if ai_progress['epoch']:
                            print(f"  Epoch: {ai_progress['epoch']:.1f}/3.0")
                        if ai_progress['accuracy']:
                            print(f"  Validation Accuracy: {ai_progress['accuracy']:.4f}")
                        if ai_progress['loss']:
                            print(f"  Validation Loss: {ai_progress['loss']:.4f}")
                    else:
                        status = ai_progress.get('status', 'unknown')
                        if status == 'starting':
                            print("  Status: Training starting...")
                        else:
                            print(f"  Status: {status}")
                else:
                    print("  Status: No training detected or training stopped")
            print()
            
            print("=" * 80)
            print(f"Refreshing in 3 seconds... (Ctrl+C to stop)")
            print()
            
            time.sleep(3)
            print("\033[2J\033[H")  # Clear screen for next update
            print("=" * 80)
            print("LIVE TRAINING PROGRESS MONITOR")
            print("=" * 80)
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("Press Ctrl+C to stop")
            print()
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    monitor_live()
