import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# 1. Synthetic Data
data = [
    ("hello world", 1),
    ("goodbye world", 0),
    ("hello everyone", 1),
    ("goodbye everyone", 0)
]

# 2. Tokenization - simple split on space
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.word_to_idx = self.build_vocab(data)
    
    def build_vocab(self, data):
        words = set()
        for sentence, _ in data:
            words.update(sentence.split())  # Split each sentence into words
        word_to_idx = {word: idx for idx, word in enumerate(words)}
        return word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        tokenized = [self.word_to_idx[word] for word in sentence.split()]
        return torch.tensor(tokenized, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# 3. Simple Embedding-based Classifier
class TextClassifier(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim=10):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)  # Average embeddings for simplicity
        return self.fc(embedded).squeeze()

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        outputs = self(tokens)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# 4. DataLoader
dataset = TextDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Model Training
model = TextClassifier(vocab_size=len(dataset.word_to_idx))
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1)

print("Training started...")
trainer.fit(model, dataloader)
print("Training completed!")

# 6. Model Testing (on new data)
test_sentence = "hello world"
test_tokens = torch.tensor([dataset.word_to_idx[word] for word in test_sentence.split()], dtype=torch.long)
model.eval()
with torch.no_grad():
    prediction = model(test_tokens.unsqueeze(0))  # Unsqueeze to add batch dimension
    print(f"\nTest Prediction for '{test_sentence}':", torch.sigmoid(prediction).item())

