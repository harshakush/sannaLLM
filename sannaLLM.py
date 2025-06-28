import torch
import torch.nn as nn
import torch.optim as optim
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

def strip_pos(word):
    return word.split('/')[0]  # Remove POS tag if present

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

def words_to_indices(words, word_to_idx):
    return [word_to_idx.get(w, word_to_idx['<unk>']) for w in words]

def make_ngram_dataset(sentences, ngram, word_to_idx):
    inputs, targets = [], []
    for words in sentences:
        if len(words) < ngram + 1:
            continue
        idxs = words_to_indices(words, word_to_idx)
        for i in range(len(idxs) - ngram):
            context = idxs[i:i+ngram]
            target = idxs[i+ngram]
            inputs.append(context)
            targets.append(target)
    return np.array(inputs), np.array(targets)

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

    # 2. Model setup
    embedding_dim = 128
    hidden_dim = 256
    output_dim = vocab_size
    ngram = 3
    batch_size = 128

    model = NgramWordLM(vocab_size, embedding_dim, hidden_dim, output_dim, ngram=ngram).to(device)  # <-- Move model to device
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params}")

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

    # 4. Prepare datasets
    print("Preparing training and validation datasets...")
    train_inputs, train_targets = make_ngram_dataset(train_sentences, ngram, word_to_idx)
    val_inputs, val_targets = make_ngram_dataset(val_sentences, ngram, word_to_idx)
    print(f"Train samples: {len(train_inputs)}, Validation samples: {len(val_inputs)}")

    # 5. Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} starting...")
        start_time = time.time()

        perm = np.random.permutation(len(train_inputs))
        train_inputs = train_inputs[perm]
        train_targets = train_targets[perm]
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(train_inputs), batch_size), desc=f"Training Epoch {epoch+1}"):
            batch_in = torch.tensor(train_inputs[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
            batch_tg = torch.tensor(train_targets[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
            optimizer.zero_grad()
            outputs = model(batch_in)
            loss = criterion(outputs, batch_tg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_in.size(0)
        avg_train_loss = total_loss / len(train_inputs)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in tqdm(range(0, len(val_inputs), batch_size), desc=f"Validation Epoch {epoch+1}"):
                batch_in = torch.tensor(val_inputs[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
                batch_tg = torch.tensor(val_targets[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
                outputs = model(batch_in)
                loss = criterion(outputs, batch_tg)
                val_loss += loss.item() * batch_in.size(0)
            avg_val_loss = val_loss / len(val_inputs)
            val_losses.append(avg_val_loss)
        end_time = time.time()
        print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds.")
        print(f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

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

    # 6. Save vocabulary and training log
    with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump({'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word, 'vocab': vocab}, f)
    print("Vocabulary saved.")

    with open(os.path.join(output_dir, "training_log.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, val) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i+1, tr, val])
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

    # 9. Interactive generation
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

# === RESUME TRAINING ===
elif mode == "resume":
    output_dir = resume_dir
    print(f"Resuming from: {output_dir}")

    # 1. Load hyperparameters, vocab, model, optimizer
    with open(os.path.join(output_dir, "model_hyperparams.json")) as f:
        hparams = json.load(f)
    with open(os.path.join(output_dir, "vocab.pkl"), "rb") as f:
        vocab_data = pickle.load(f)
    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = vocab_data['idx_to_word']
    vocab = vocab_data['vocab']

    embedding_dim = hparams["embedding_dim"]
    hidden_dim = hparams["hidden_dim"]
    output_dim = hparams["output_dim"]
    ngram = hparams["ngram"]
    vocab_size = hparams["vocab_size"]
    batch_size = hparams["batch_size"]

    model = NgramWordLM(vocab_size, embedding_dim, hidden_dim, output_dim, ngram=ngram).to(device)  # <-- To device
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_ngram_model.pt"), map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(torch.load(os.path.join(output_dir, "optimizer.pt"), map_location=device))
    criterion = nn.CrossEntropyLoss()

    # 2. Reload and preprocess your data as before
    all_filenames = [f for f in glob.glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]
    random.shuffle(all_filenames)
    word_counter = Counter()
    sentences = []
    for filename in tqdm(all_filenames, desc="Reading files"):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if " " in line and any(c.isalpha() for c in line):
                    words = ['<s>'] + [strip_pos(w.strip(string.punctuation)) for w in line.lower().split() if w.strip(string.punctuation)] + ['</s>']
                    word_counter.update(words)
                    sentences.append(words)
    split_idx = int(0.9 * len(sentences))
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    train_inputs, train_targets = make_ngram_dataset(train_sentences, ngram, word_to_idx)
    val_inputs, val_targets = make_ngram_dataset(val_sentences, ngram, word_to_idx)

    # 3. Resume training
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    print("\nResuming training...")
    for epoch in range(resume_epochs):
        print(f"\nResumed Epoch {epoch+1}/{resume_epochs} starting...")
        start_time = time.time()

        perm = np.random.permutation(len(train_inputs))
        train_inputs = train_inputs[perm]
        train_targets = train_targets[perm]
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(train_inputs), batch_size), desc=f"Training Epoch {epoch+1}"):
            batch_in = torch.tensor(train_inputs[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
            batch_tg = torch.tensor(train_targets[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
            optimizer.zero_grad()
            outputs = model(batch_in)
            loss = criterion(outputs, batch_tg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_in.size(0)
        avg_train_loss = total_loss / len(train_inputs)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in tqdm(range(0, len(val_inputs), batch_size), desc=f"Validation Epoch {epoch+1}"):
                batch_in = torch.tensor(val_inputs[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
                batch_tg = torch.tensor(val_targets[i:i+batch_size], dtype=torch.long, device=device)  # <-- To device
                outputs = model(batch_in)
                loss = criterion(outputs, batch_tg)
                val_loss += loss.item() * batch_in.size(0)
            avg_val_loss = val_loss / len(val_inputs)
            val_losses.append(avg_val_loss)
        end_time = time.time()
        print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds.")
        print(f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

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

    # Optionally, append to your training log as before
    with open(os.path.join(output_dir, "training_log.csv"), "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, (tr, val) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([f"resumed_{i+1}", tr, val])
    print("Resumed training complete and log updated.")

else:
    print("Set mode to 'train' or 'resume'.")
