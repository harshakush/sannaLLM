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

def strip_pos(word):
    return word.split('/')[0]  # Remove POS tag if present

folder_path = r"C:\Dev\Genai\sannaLLM\brown"
print(f"Loading files from: {folder_path}")
all_filenames = [f for f in glob.glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]
print(f"Found {len(all_filenames)} files.")
random.shuffle(all_filenames)

# 1. Build vocabulary from all files, count word frequencies
word_counter = Counter()
sentences = []
print("Building vocabulary and loading sentences...")
for filename in tqdm(all_filenames, desc="Reading files"):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if " " in line and any(c.isalpha() for c in line):
                # Add sentence boundary tokens
                words = ['<s>'] + [strip_pos(w.strip(string.punctuation)) for w in line.lower().split() if w.strip(string.punctuation)] + ['</s>']
                word_counter.update(words)
                sentences.append(words)
print(f"Total sentences loaded: {len(sentences)}")
print(f"Total unique words (before filtering): {len(word_counter)}")

# 2. Filter vocabulary: keep words that appear at least 5 times
min_word_freq = 5
vocab = [w for w, c in word_counter.items() if c >= min_word_freq]
if '<unk>' not in vocab:
    vocab.append('<unk>')
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

print(f"Total unique words in vocabulary (min freq {min_word_freq}): {vocab_size}")

# 3. Split into train/validation files (90% train, 10% val)
split_idx = int(0.9 * len(sentences))
train_sentences = sentences[:split_idx]
val_sentences = sentences[split_idx:]
print(f"Train sentences: {len(train_sentences)}, Validation sentences: {len(val_sentences)}")

# 4. Model setup: 3-gram feedforward
embedding_dim = 128
hidden_dim = 256
output_dim = vocab_size
ngram = 3
batch_size = 128

class NgramWordLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, ngram=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.ngram = ngram
        self.fc1 = nn.Linear(embedding_dim * ngram, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x shape: (batch_size, ngram)
        x = self.embedding(x)  # (batch_size, ngram, embedding_dim)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, ngram * embedding_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NgramWordLM(vocab_size, embedding_dim, hidden_dim, output_dim, ngram=ngram)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in model: {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Helper: Convert words to indices, use <unk> for OOV
def words_to_indices(words):
    return [word_to_idx.get(w, word_to_idx['<unk>']) for w in words]

# 6. Helper: Create n-gram dataset from sentences
def make_ngram_dataset(sentences, ngram):
    inputs, targets = [], []
    for words in sentences:
        if len(words) < ngram + 1:
            continue
        idxs = words_to_indices(words)
        for i in range(len(idxs) - ngram):
            context = idxs[i:i+ngram]
            target = idxs[i+ngram]
            inputs.append(context)
            targets.append(target)
    return np.array(inputs), np.array(targets)

print("Preparing training and validation datasets...")
train_inputs, train_targets = make_ngram_dataset(train_sentences, ngram)
val_inputs, val_targets = make_ngram_dataset(val_sentences, ngram)
print(f"Train samples: {len(train_inputs)}, Validation samples: {len(val_inputs)}")

# 8. Training loop with mini-batching and early stopping
num_epochs = 10
patience = 2
best_val_loss = float('inf')
epochs_no_improve = 0

print("\nStarting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs} starting...")
    start_time = time.time()

    # Shuffle training data
    perm = np.random.permutation(len(train_inputs))
    train_inputs = train_inputs[perm]
    train_targets = train_targets[perm]
    model.train()
    total_loss = 0

    # Progress bar for training batches
    for i in tqdm(range(0, len(train_inputs), batch_size), desc=f"Training Epoch {epoch+1}"):
        batch_in = torch.tensor(train_inputs[i:i+batch_size], dtype=torch.long)
        batch_tg = torch.tensor(train_targets[i:i+batch_size], dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(batch_in)
        loss = criterion(outputs, batch_tg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_in.size(0)
    avg_train_loss = total_loss / len(train_inputs)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for i in tqdm(range(0, len(val_inputs), batch_size), desc=f"Validation Epoch {epoch+1}"):
            batch_in = torch.tensor(val_inputs[i:i+batch_size], dtype=torch.long)
            batch_tg = torch.tensor(val_targets[i:i+batch_size], dtype=torch.long)
            outputs = model(batch_in)
            loss = criterion(outputs, batch_tg)
            val_loss += loss.item() * batch_in.size(0)
        avg_val_loss = val_loss / len(val_inputs)
    end_time = time.time()
    print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds.")
    print(f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_ngram_model.pt")
        print("Validation loss improved, model saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Load best model
print("Loading best model from disk...")
model.load_state_dict(torch.load("best_ngram_model.pt"))

# 9. Generate a sequence of words
def generate_sequence(seed_words, num_words=20, sampling=True, temperature=1.0):
    generated = list(seed_words)
    current_words = list(seed_words)
    for _ in range(num_words):
        input_idx = torch.tensor([[word_to_idx.get(w, word_to_idx['<unk>']) for w in current_words]], dtype=torch.long)
        output = model(input_idx)
        probs = torch.softmax(output / temperature, dim=1).detach().numpy().flatten()
        if sampling:
            next_idx = torch.multinomial(torch.tensor(probs), 1).item()
        else:
            next_idx = torch.argmax(output).item()
        next_word = idx_to_word[next_idx]
        generated.append(next_word)
        current_words = current_words[1:] + [next_word]
    return ' '.join(generated)

# 10. Prompt user for seed words and generate text until 'exit' is entered
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
