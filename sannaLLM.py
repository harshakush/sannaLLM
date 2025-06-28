import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import random
import string

def strip_pos(word):
    return word.split('/')[0]  # Remove POS tag if present

folder_path = r"C:\Dev\Genai\sannaLLM\brown"
all_filenames = [f for f in glob.glob(os.path.join(folder_path, "*")) if os.path.isfile(f)]

# 1. Build vocabulary from all files
vocab_set = set()
max_vocab_files = len(all_filenames)
for filename in all_filenames[:max_vocab_files]:
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if " " in line and any(c.isalpha() for c in line):
                for w in line.lower().split():
                    w = w.strip(string.punctuation)
                    w = strip_pos(w)
                    if w:
                        vocab_set.add(w)
vocab = list(vocab_set)
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

print(f"Total unique words in vocabulary: {vocab_size}")

# 2. Model setup: 3-gram feedforward
embedding_dim = 64
hidden_dim = 128
output_dim = vocab_size
ngram = 3

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
print(f"Total parameters: {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Train on each file sequentially
num_epochs_per_file = 1  # You can increase this for more thorough training
for file_idx, filename in enumerate(all_filenames):
    file_words = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if " " in line and any(c.isalpha() for c in line):
                for w in line.lower().split():
                    w = w.strip(string.punctuation)
                    w = strip_pos(w)
                    if w and w in word_to_idx:
                        file_words.append(w)
    if len(file_words) < ngram + 1:
        continue  # Skip files that are too short
    # Prepare n-gram training data
    inputs = []
    targets = []
    for i in range(len(file_words) - ngram):
        context = file_words[i:i+ngram]
        target = file_words[i+ngram]
        inputs.append([word_to_idx[w] for w in context])
        targets.append(word_to_idx[target])
    if not inputs:
        continue
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    for epoch in range(num_epochs_per_file):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (file_idx + 1) % 10 == 0 or file_idx == len(all_filenames) - 1:
        print(f"Trained on file {file_idx + 1} of {len(all_filenames)} (Loss: {loss.item():.4f})")

# 4. Generate a sequence of words
def generate_sequence(seed_words, num_words=20, sampling=True, temperature=1.0):
    generated = list(seed_words)
    current_words = list(seed_words)
    for _ in range(num_words):
        input_idx = torch.tensor([[word_to_idx.get(w, 0) for w in current_words]], dtype=torch.long)
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

# 5. Prompt user for seed words and generate text until 'exit' is entered
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
    generated_text = generate_sequence(seed_words, num_words=30, sampling=True, temperature=1.0)
    print(f"\nGenerated sequence:\n{generated_text}\n")
