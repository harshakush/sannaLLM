import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define a small vocabulary
vocab = [
    "the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "slow", "bird",
    "flew", "high", "low", "fish", "swam", "deep", "blue", "sky", "green", "grass", "is"
]
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# 2. Create 8 samples of 3-grams (as word lists)
samples = [
    ["the", "cat", "sat"],
    ["the", "dog", "ran"],
    ["the", "bird", "flew"],
    ["the", "fish", "swam"],
    ["the", "sky", "is"],
    ["the", "grass", "is"],
    ["the", "cat", "ran"],
    ["the", "dog", "sat"]
]

# 3. Map words to indices
inputs = torch.tensor([[word_to_idx[w] for w in sample] for sample in samples], dtype=torch.long)
print("Input 3-grams as words:")
for row in samples:
    print(row)
print("\nInput indices:\n", inputs)

# 4. Define the 7-layer network
embedding_dim = 8
hidden_dim = 16
output_dim = 20
ngram = 3
batch_size = 8

class SevenLayerNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, ngram):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * ngram, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Layer 1: Embedding (non-linear lookup)
        x = self.embedding(x)
        # Flatten for linear layers
        x = x.view(x.size(0), -1)
        # Layer 2: Linear
        x = self.fc1(x)
        # Layer 3: ReLU (non-linear)
        x = F.relu(x)
        # Layer 4: Linear
        x = self.fc2(x)
        # Layer 5: ReLU (non-linear)
        x = F.relu(x)
        # Layer 6: Linear
        x = self.fc3(x)
        # Layer 7: Output Linear
        x = self.fc5(x)
        return x

# 5. Instantiate model
model = SevenLayerNet(vocab_size, embedding_dim, hidden_dim, output_dim, ngram)

# 6. Display initial weights and bias of the output layer
print("\nInitial output layer weights (fc5):\n", model.fc5.weight.data)
print("Initial output layer bias (fc5):\n", model.fc5.bias.data)

# 7. Dummy targets for training (random for demonstration)
targets = torch.randn(batch_size, output_dim)

# 8. Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 9. Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# 10. Display trained weights and bias of the output layer
print("\nTrained output layer weights (fc5):\n", model.fc5.weight.data)
print("Trained output layer bias (fc5):\n", model.fc5.bias.data)

# 11. Final output
output = model(inputs)
print("\nFinal output shape:", output.shape)
