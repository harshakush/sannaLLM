import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import glob
import os
import random
import string
from collections import Counter
import numpy as np
import time
from tqdm import tqdm
import pickle
import json
import csv
from datetime import datetime
import psutil

# === CPU OPTIMIZATION SETTINGS ===
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

def strip_pos(word):
    return word.split('/')[0]  # Remove POS tag if present

# === CUSTOM DATASET CLASS ===
class NgramDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# === MODEL DEFINITION ===
class NgramWordLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, ngram=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.ngram = ngram
        self.fc1 = nn.Linear(embedding_dim * ngram, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def make_ngram_dataset_fast(sentences, ngram, word_to_idx):
    """Optimized version of make_ngram_dataset"""
    all_inputs, all_targets = [], []
    
    for words in sentences:
        if len(words) < ngram + 1:
            continue
        idxs = [word_to_idx.get(w, word_to_idx['<unk>']) for w in words]
        
        # Vectorized approach
        for i in range(len(idxs) - ngram):
            all_inputs.append(idxs[i:i+ngram])
            all_targets.append(idxs[i+ngram])
    
    return np.array(all_inputs, dtype=np.int64), np.array(all_targets, dtype=np.int64)

def monitor_resources():
    """Monitor system resources"""
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

def main():
    # === USER SETTINGS ===
    mode = "train"  # "train" or "resume"
    resume_dir = "ngram_output_YYYYMMDD_HHMMSS"  # Set this if mode == "resume"
    folder_path = r"C:\Dev\Genai\sannaLLM\brown"
    num_epochs = 10  # For training from scratch
    resume_epochs = 5  # For resuming
    patience = 2

    # === DEVICE SETUP ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available CPU cores: {os.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    # === TRAINING FROM SCRATCH ===
    if mode == "train":
        # 0. Output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"ngram_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"All outputs will be saved in: {output_dir}")

        # 1. Data loading and vocabulary
        print(f"Loading files from: {folder_path}")
        all_filenames = [f for f in glob.glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]
        print(f"Found {len(all_filenames)} files.")
        random.shuffle(all_filenames)

        word_counter = Counter()
        sentences = []
        print("Building vocabulary and loading sentences...")
        for filename in tqdm(all_filenames, desc="Reading files"):
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if " " in line and any(c.isalpha() for c in line):
                        words = ['<s>'] + [strip_pos(w.strip(string.punctuation)) for w in line.lower().split() if w.strip(string.punctuation)] + ['</s>']
                        word_counter.update(words)
                        sentences.append(words)
        print(f"Total sentences loaded: {len(sentences)}")
        print(f"Total unique words (before filtering): {len(word_counter)}")

        min_word_freq = 5
        vocab = [w for w, c in word_counter.items() if c >= min_word_freq]
        if '<unk>' not in vocab:
            vocab.append('<unk>')
        vocab_size = len(vocab)
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        idx_to_word = {i: w for i, w in enumerate(vocab)}
        print(f"Total unique words in vocabulary (min freq {min_word_freq}): {vocab_size}")

        split_idx = int(0.9 * len(sentences))
        train_sentences = sentences[:split_idx]
        val_sentences = sentences[split_idx:]
        print(f"Train sentences: {len(train_sentences)}, Validation sentences: {len(val_sentences)}")

        # 2. Model setup with optimized batch size
        embedding_dim = 128
        hidden_dim = 256
        output_dim = vocab_size
        ngram = 3
        batch_size = 512  # Increased from 128 for better performance

        model = NgramWordLM(vocab_size, embedding_dim, hidden_dim, output_dim, ngram=ngram).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in model: {total_params}")
        print(f"Model is on device: {next(model.parameters()).device}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 3. Save hyperparameters
        hyperparams = {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "ngram": ngram,
            "vocab_size": vocab_size,
            "batch_size": batch_size
        }
        with open(os.path.join(output_dir, "model_hyperparams.json"), "w") as f:
            json.dump(hyperparams, f)
        print("Hyperparameters saved.")

        # 4. Prepare datasets with Windows-compatible DataLoader
        print("Preparing training and validation datasets...")
        train_inputs, train_targets = make_ngram_dataset_fast(train_sentences, ngram, word_to_idx)
        val_inputs, val_targets = make_ngram_dataset_fast(val_sentences, ngram, word_to_idx)
        print(f"Train samples: {len(train_inputs)}, Validation samples: {len(val_inputs)}")

        # Create datasets and dataloaders - WINDOWS FIX: num_workers=0
        train_dataset = NgramDataset(train_inputs, train_targets)  # Fixed: correct targets
        val_dataset = NgramDataset(val_inputs, val_targets)

        # WINDOWS COMPATIBILITY: Use single-threaded DataLoader
        print("Using single-threaded DataLoader for Windows compatibility")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Windows fix: no multiprocessing
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Windows fix: no multiprocessing
            pin_memory=True if device.type == 'cuda' else False
        )

        # 5. Enhanced training loop with better time tracking
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        epoch_times = []

        print("\nStarting training...")
        print("Initial resource usage:")
        monitor_resources()
        
        training_start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} starting...")
            epoch_start_time = time.time()

            # Training
            model.train()
            total_loss = 0
            num_samples = 0

            for batch_in, batch_tg in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                batch_in = batch_in.to(device, non_blocking=True)
                batch_tg = batch_tg.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(batch_in)
                loss = criterion(outputs, batch_tg)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_in.size(0)
                num_samples += batch_in.size(0)

            avg_train_loss = total_loss / num_samples
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0
            val_samples = 0
            
            with torch.no_grad():
                for batch_in, batch_tg in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                    batch_in = batch_in.to(device, non_blocking=True)
                    batch_tg = batch_tg.to(device, non_blocking=True)
                    
                    outputs = model(batch_in)
                    loss = criterion(outputs, batch_tg)
                    val_loss += loss.item() * batch_in.size(0)
                    val_samples += batch_in.size(0)

            avg_val_loss = val_loss / val_samples
            val_losses.append(avg_val_loss)

            # Enhanced time tracking
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            
            # Calculate remaining time estimate
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            epochs_remaining = num_epochs - (epoch + 1)
            estimated_time_remaining = avg_epoch_time * epochs_remaining
            
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")
            print(f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
            if epochs_remaining > 0:
                hours = int(estimated_time_remaining // 3600)
                minutes = int((estimated_time_remaining % 3600) // 60)
                seconds = int(estimated_time_remaining % 60)
                print(f"Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Monitor resources every few epochs
            if (epoch + 1) % 3 == 0:
                print("Resource usage:")
                monitor_resources()

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(output_dir, "best_ngram_model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                print("Validation loss improved, model and optimizer saved.")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        total_training_time = time.time() - training_start_time
        print(f"\nTotal training completed in {total_training_time/60:.2f} minutes.")

        # 6. Save vocabulary and training log
        with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
            pickle.dump({'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word, 'vocab': vocab}, f)
        print("Vocabulary saved.")

        with open(os.path.join(output_dir, "training_log.csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "val_loss", "epoch_time"])
            for i, (tr, val, time_taken) in enumerate(zip(train_losses, val_losses, epoch_times)):
                writer.writerow([i+1, tr, val, time_taken])
        print("Training log saved.")

        # 7. Load best model for inference
        print("Loading best model from disk...")
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_ngram_model.pt"), map_location=device))
        model.eval()

        # 8. Generation function
        def generate_sequence(seed_words, num_words=20, sampling=True, temperature=1.0):
            generated = list(seed_words)
            current_words = list(seed_words)
            for _ in range(num_words):
                input_idx = torch.tensor([[word_to_idx.get(w, word_to_idx['<unk>']) for w in current_words]], dtype=torch.long, device=device)
                output = model(input_idx)
                probs = torch.softmax(output / temperature, dim=1).detach().cpu().numpy().flatten()
                if sampling:
                    next_idx = torch.multinomial(torch.tensor(probs), 1).item()
                else:
                    next_idx = torch.argmax(output).item()
                next_word = idx_to_word[next_idx]
                generated.append(next_word)
                current_words = current_words[1:] + [next_word]
            return ' '.join(generated)

        # 9. Auto-generate some sample text
        print("\nAuto-generating sample text...")
        sample_seeds = [['the', 'quick', 'brown'], ['in', 'the', 'beginning'], ['it', 'was', 'a']]
        for seed in sample_seeds:
            if all(w in word_to_idx for w in seed):
                text = generate_sequence(seed, num_words=20)
                print(f"Seed: {' '.join(seed)} -> {text}")

        # 10. Interactive generation
        while True:
            seed_input = input(f"\nEnter {ngram} seed words separated by spaces (or type 'sample' to see some options, or 'exit' to quit): ").strip().lower()
            if seed_input == 'exit':
                print("Exiting.")
                break
            if seed_input == 'sample':
                print("Sample vocabulary words:", random.sample(vocab, min(20, len(vocab))))
                continue
            seed_words = [w.strip(string.punctuation) for w in seed_input.split()]
            if len(seed_words) != ngram:
                print(f"Please enter exactly {ngram} seed words.")
                continue
            if not all(w in word_to_idx for w in seed_words):
                print("One or more seed words are not in the vocabulary. Try again or type 'sample'.")
                continue
            print("Generating sequence, please wait...")
            generated_text = generate_sequence(seed_words, num_words=30, sampling=True, temperature=1.0)
            print(f"\nGenerated sequence:\n{generated_text}\n")

    else:
        print("Set mode to 'train' or 'resume'.")

# WINDOWS COMPATIBILITY: Main guard
if __name__ == '__main__':
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
