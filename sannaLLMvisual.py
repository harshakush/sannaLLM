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

# === VISUALIZER IMPORT ===
from nn_visualizer import NeuralNetworkVisualizer

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
    all_inputs, all_targets = [], []
    for words in sentences:
        if len(words) < ngram + 1:
            continue
        idxs = [word_to_idx.get(w, word_to_idx['<unk>']) for w in words]
        for i in range(len(idxs) - ngram):
            all_inputs.append(idxs[i:i+ngram])
            all_targets.append(idxs[i+ngram])
    return np.array(all_inputs, dtype=np.int64), np.array(all_targets, dtype=np.int64)

def monitor_resources():
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

def calculate_perplexity(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch_in, batch_tg in data_loader:
            batch_in = batch_in.to(device, non_blocking=True)
            batch_tg = batch_tg.to(device, non_blocking=True)
            outputs = model(batch_in)
            loss = criterion(outputs, batch_tg)
            total_loss += loss.item() * batch_in.size(0)
            total_samples += batch_in.size(0)
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def main():
    # === USER SETTINGS ===
    mode = "train"  # "train" or "resume"
    resume_dir = "ngram_output_YYYYMMDD_HHMMSS"  # Set this if mode == "resume"
    folder_path = r"C:\Dev\Genai\sannaLLM\brown"
    num_epochs = 10  # 2 For training from scratch, 10 for regular
    resume_epochs = 5  # For resuming
    
    # === IMPROVED EARLY STOPPING SETTINGS ===
    patience = 8 #2 for testing 8 for regular 
    min_epochs = 10
    improvement_threshold = 1e-4

    # === DEVICE SETUP ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available CPU cores: {os.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"PyTorch version: {torch.__version__}")

    if mode == "train":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"ngram_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"All outputs will be saved in: {output_dir}")

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

        embedding_dim = 128
        hidden_dim = 256
        output_dim = vocab_size
        ngram = 3
        batch_size = 3084

        model = NgramWordLM(vocab_size, embedding_dim, hidden_dim, output_dim, ngram=ngram).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in model: {total_params}")
        print(f"Model is on device: {next(model.parameters()).device}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        print("Learning rate scheduler initialized")

        hyperparams = {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "ngram": ngram,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "patience": patience,
            "min_epochs": min_epochs,
            "improvement_threshold": improvement_threshold,
            "pytorch_version": torch.__version__
        }
        with open(os.path.join(output_dir, "model_hyperparams.json"), "w") as f:
            json.dump(hyperparams, f, indent=2)
        print("Hyperparameters saved.")

        print("Preparing training and validation datasets...")
        train_inputs, train_targets = make_ngram_dataset_fast(train_sentences, ngram, word_to_idx)
        val_inputs, val_targets = make_ngram_dataset_fast(val_sentences, ngram, word_to_idx)
        print(f"Train samples: {len(train_inputs)}, Validation samples: {len(val_inputs)}")

        train_dataset = NgramDataset(train_inputs, train_targets)
        val_dataset = NgramDataset(val_inputs, val_targets)

        print("Using single-threaded DataLoader for Windows compatibility")
        ''' train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        '''
        train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # Increased batch size
        shuffle=True, 
        num_workers=32,    # More workers
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=32 # More batches prefetched per worker
        )

        val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # Increased batch size
        shuffle=False, 
        num_workers=32,    # More workers
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=32 # More batches prefetched per worker
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        epoch_times = []
        perplexities = []
        learning_rates = []

        print(f"\nStarting training with improved early stopping:")
        print(f"- Patience: {patience} epochs")
        print(f"- Minimum epochs before early stopping: {min_epochs}")
        print(f"- Improvement threshold: {improvement_threshold}")
        print("Initial resource usage:")
        monitor_resources()
        
        training_start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} starting...")
            epoch_start_time = time.time()

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
            perplexity = calculate_perplexity(model, val_loader, criterion, device)
            perplexities.append(perplexity)

            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            if current_lr < old_lr:
                print(f"Learning rate reduced: {old_lr:.2e} -> {current_lr:.2e}")

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            epochs_remaining = num_epochs - (epoch + 1)
            estimated_time_remaining = avg_epoch_time * epochs_remaining

            print(f"\nEpoch {epoch+1} Results:")
            print(f"- Duration: {epoch_duration:.2f} seconds")
            print(f"- Train Loss: {avg_train_loss:.6f}")
            print(f"- Val Loss: {avg_val_loss:.6f}")
            print(f"- Perplexity: {perplexity:.2f}")
            print(f"- Learning Rate: {current_lr:.2e}")
            print(f"- Best Val Loss so far: {best_val_loss:.6f}")
            if epochs_remaining > 0:
                hours = int(estimated_time_remaining // 3600)
                minutes = int((estimated_time_remaining % 3600) // 60)
                seconds = int(estimated_time_remaining % 60)
                print(f"- Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
            if (epoch + 1) % 3 == 0:
                print("\nResource usage:")
                monitor_resources()

            improvement = best_val_loss - avg_val_loss
            if improvement > improvement_threshold:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(output_dir, "best_ngram_model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                print(f"✓ Validation loss improved by {improvement:.6f}, model saved.")
            else:
                epochs_no_improve += 1
                print(f"✗ No significant improvement for {epochs_no_improve} epoch(s) (threshold: {improvement_threshold:.6f})")
                if epochs_no_improve >= patience and epoch >= min_epochs:
                    print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
                    print(f"   - No improvement for {epochs_no_improve} consecutive epochs")
                    print(f"   - Minimum epochs requirement ({min_epochs}) satisfied")
                    break
                elif epoch < min_epochs:
                    print(f"   - Still in minimum epochs phase ({epoch+1}/{min_epochs})")

        total_training_time = time.time() - training_start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_training_time/60:.2f} minutes")
        print(f"Final best validation loss: {best_val_loss:.6f}")
        if perplexities:
            print(f"Final perplexity: {perplexities[-1]:.2f}")

        with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
            pickle.dump({'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word, 'vocab': vocab}, f)
        print("Vocabulary saved.")

        with open(os.path.join(output_dir, "training_log.csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "val_loss", "perplexity", "epoch_time", "learning_rate"])
            for i, (tr, val, perp, time_taken, lr) in enumerate(zip(train_losses, val_losses, perplexities, epoch_times, learning_rates)):
                writer.writerow([i+1, tr, val, perp, time_taken, lr])
        print("Enhanced training log saved.")

        print("\nLoading best model from disk...")
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_ngram_model.pt"), map_location=device))
        model.eval()

        # === VISUALIZATION BLOCK ===
        print("\nGenerating model visualizations...")
        visualizer = NeuralNetworkVisualizer(model, word_to_idx, idx_to_word, output_dir=output_dir)
        visualizer.visualize_all(sample_words=['the', 'quick', 'brown'])
        print("Visualizations saved in:", output_dir)
        # === END VISUALIZATION BLOCK ===

        def generate_sequence(seed_words, num_words=20, sampling=True, temperature=1.0, top_k=None):
            generated = list(seed_words)
            current_words = list(seed_words)
            for _ in range(num_words):
                input_idx = torch.tensor([[word_to_idx.get(w, word_to_idx['<unk>']) for w in current_words]], dtype=torch.long, device=device)
                with torch.no_grad():
                    output = model(input_idx)
                    logits = output / temperature
                    if top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        probs = torch.softmax(top_k_logits, dim=1)
                        next_idx_pos = torch.multinomial(probs, 1).item()
                        next_idx = top_k_indices[0][next_idx_pos].item()
                    elif sampling:
                        probs = torch.softmax(logits, dim=1)
                        next_idx = torch.multinomial(probs, 1).item()
                    else:
                        next_idx = torch.argmax(logits).item()
                next_word = idx_to_word[next_idx]
                if next_word == '</s>':
                    break
                generated.append(next_word)
                current_words = current_words[1:] + [next_word]
            return ' '.join(generated)

        print(f"\n{'='*60}")
        print("Auto-generating sample text with different strategies...")
        sample_seeds = [['the', 'quick', 'brown'], ['in', 'the', 'beginning'], ['it', 'was', 'a'], ['once', 'upon', 'a']]
        for seed in sample_seeds:
            if all(w in word_to_idx for w in seed):
                print(f"\nSeed: '{' '.join(seed)}'")
                text_greedy = generate_sequence(seed, num_words=15, sampling=False)
                print(f"  Greedy: {text_greedy}")
                text_temp = generate_sequence(seed, num_words=15, sampling=True, temperature=0.8)
                print(f"  Temp=0.8: {text_temp}")
                text_topk = generate_sequence(seed, num_words=15, sampling=True, temperature=1.0, top_k=10)
                print(f"  Top-k=10: {text_topk}")

        print(f"\n{'='*60}")
        print("Interactive Text Generation")
        print("Commands:")
        print("  - Enter 3 seed words separated by spaces")
        print("  - Type 'sample' to see vocabulary examples")
        print("  - Type 'stats' to see model statistics")
        print("  - Type 'exit' to quit")
        while True:
            seed_input = input(f"\nEnter {ngram} seed words: ").strip().lower()
            if seed_input == 'exit':
                print("Exiting. Goodbye!")
                break
            if seed_input == 'sample':
                print("Sample vocabulary words:", random.sample(vocab, min(20, len(vocab))))
                continue
            if seed_input == 'stats':
                print(f"Model Statistics:")
                print(f"  - Vocabulary size: {vocab_size:,}")
                print(f"  - Training samples: {len(train_inputs):,}")
                print(f"  - Validation samples: {len(val_inputs):,}")
                print(f"  - Model parameters: {total_params:,}")
                if perplexities:
                    print(f"  - Final perplexity: {perplexities[-1]:.2f}")
                continue
            seed_words = [w.strip(string.punctuation) for w in seed_input.split()]
            if len(seed_words) != ngram:
                print(f"❌ Please enter exactly {ngram} seed words.")
                continue
            if not all(w in word_to_idx for w in seed_words):
                missing_words = [w for w in seed_words if w not in word_to_idx]
                print(f"❌ Words not in vocabulary: {missing_words}")
                print("Try 'sample' to see available words.")
                continue
            print("🔄 Generating sequences...")
            try:
                greedy = generate_sequence(seed_words, num_words=25, sampling=False)
                creative = generate_sequence(seed_words, num_words=25, sampling=True, temperature=1.2)
                balanced = generate_sequence(seed_words, num_words=25, sampling=True, temperature=0.8)
                print(f"\n📝 Generated Text:")
                print(f"  Conservative: {greedy}")
                print(f"  Balanced: {balanced}")
                print(f"  Creative: {creative}")
            except Exception as e:
                print(f"❌ Generation error: {e}")

    elif mode == "resume":
        print(f"Resuming training from: {resume_dir}")
        if not os.path.exists(resume_dir):
            print(f"❌ Resume directory not found: {resume_dir}")
            return
        vocab_path = os.path.join(resume_dir, "vocab.pkl")
        if not os.path.exists(vocab_path):
            print(f"❌ Vocabulary file not found: {vocab_path}")
            return
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)
            word_to_idx = vocab_data['word_to_idx']
            idx_to_word = vocab_data['idx_to_word']
            vocab = vocab_data['vocab']
        print(f"✓ Loaded vocabulary with {len(vocab)} words")
        hyperparams_path = os.path.join(resume_dir, "model_hyperparams.json")
        if not os.path.exists(hyperparams_path):
            print(f"❌ Hyperparameters file not found: {hyperparams_path}")
            return
        with open(hyperparams_path, "r") as f:
            hyperparams = json.load(f)
        print(f"✓ Loaded hyperparameters")
        model = NgramWordLM(
            vocab_size=hyperparams['vocab_size'],
            embedding_dim=hyperparams['embedding_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            output_dim=hyperparams['output_dim'],
            ngram=hyperparams['ngram']
        ).to(device)
        model_path = os.path.join(resume_dir, "best_ngram_model.pt")
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model weights")
        print("📚 Resuming training would continue here...")
        print("(Full resume implementation would reload data and continue training)")
    else:
        print("❌ Invalid mode. Set mode to 'train' or 'resume'.")

if __name__ == '__main__':
    mp.freeze_support()
    main()
