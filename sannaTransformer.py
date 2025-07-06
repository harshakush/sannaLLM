import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
import glob
import argparse

# ========== THREADING FOR DATA LOADING ==========
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ========== DATA PREP ==========

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    vocab = [w for w, c in counter.items() if c >= min_freq]
    vocab = ['<pad>', '<unk>'] + vocab
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return vocab, word2idx, idx2word

def tokenize(sentences, word2idx):
    tokens = []
    for sent in sentences:
        tokens += [word2idx.get(w, word2idx['<unk>']) for w in sent]
    return tokens

# ========== MODEL ==========

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, emb_size=128, nhead=4, num_layers=2, dim_feedforward=256, max_seq_len=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(max_seq_len, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer(x, mask=mask)
        logits = self.fc_out(x)
        return logits

# ========== CHECKPOINTING ==========

def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}.")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pt"):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint loaded from epoch {epoch+1}.")
    return epoch, best_val_loss

def should_pause():
    return os.path.exists("PAUSE.TXT")

# ========== TRAINING LOOP ==========

def train_epoch(model, loader, optimizer, criterion, device, epoch, best_val_loss, checkpoint_path):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Train", leave=False)):
        x, y = x.to(device, non_blocking=False), y.to(device, non_blocking=False)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

        # Check for pause signal after each batch
        if should_pause():
            print("\nPause signal detected. Saving checkpoint and exiting...")
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
            print("You can now safely restart your machine. To resume, run with --resume.")
            exit(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device, non_blocking=False), y.to(device, non_blocking=False)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def calculate_perplexity(loss):
    return math.exp(loss)

# ========== MAIN SCRIPT ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pause', action='store_true', help='Pause training at the end of the next batch')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--restart', action='store_true', help='Restart training from scratch')
    args = parser.parse_args()

    # --- Hyperparameters ---
    seq_len = 32
    batch_size = 64
    emb_size = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    num_epochs = 5
    lr = 3e-4
    min_freq = 2
    max_seq_len = seq_len
    num_workers = 1
    pin_memory = False
    checkpoint_path = "checkpoint.pt"

    # --- Data Loading from local folder ---
    folder_path = r"C:\Dev\sannaLLM\brown"
    all_filenames = [f for f in glob.glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]
    print("Files found:", all_filenames)
    sentences = []
    for filename in all_filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words = [w.lower() for w in line.strip().split() if w.strip()]
                if words:
                    sentences.append(words)
    print(f"Loaded {len(sentences):,} sentences from {len(all_filenames)} files.")

    # --- Build Vocab ---
    vocab, word2idx, idx2word = build_vocab(sentences, min_freq=min_freq)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # --- Tokenize ---
    tokens = tokenize(sentences, word2idx)
    split = int(0.9 * len(tokens))
    train_tokens, val_tokens = tokens[:split], tokens[split:]

    # --- Datasets and Loaders ---
    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )

    # --- Model, Optimizer, Loss ---
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model = TransformerLM(vocab_size, emb_size, nhead, num_layers, dim_feedforward, max_seq_len, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

    # --- Checkpoint Handling ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch += 1  # Continue from next epoch
    elif args.restart:
        print("Restarting training from scratch. Previous checkpoints will be overwritten.")
        start_epoch = 0
        best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, best_val_loss, checkpoint_path)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val ppl: {calculate_perplexity(val_loss):.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer_lm.pt")
            print("Model saved.")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)

        # If --pause was passed, pause at the end of the epoch
        if args.pause:
            print("Pausing training as requested by --pause.")
            break

    print("Training complete.")

    # --- Generation Example ---
    def generate(seed, max_new_tokens=30, temperature=1.0):
        model.eval()
        tokens = [word2idx.get(w, word2idx['<unk>']) for w in seed]
        for _ in range(max_new_tokens):
            x = torch.tensor(tokens[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                logits = logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            if idx2word[next_token] == '.':
                break
        return ' '.join([idx2word[t] for t in tokens])

    print("\nSample generation:")
    print(generate(['the', 'united', 'states'], max_new_tokens=20, temperature=0.8))

if __name__ == "__main__":
    main()
