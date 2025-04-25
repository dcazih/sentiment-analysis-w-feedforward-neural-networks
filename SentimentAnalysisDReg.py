import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from FNN import FNN
import torch.nn as nn

# Load dataset
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# TF-IDF vectorization
v = TfidfVectorizer(max_features=5000)
X = v.fit_transform(df['review']).toarray()
y = df['sentiment'].values

# Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=128)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=128)

fnn = FNN(use_dropout=True, dropout_rate=0.5)
optimizer = torch.optim.AdamW(fnn.parameters(), lr=0.0001, weight_decay=0.01)
L = nn.CrossEntropyLoss()

# Training loop
epochs = 10
start = time.time()
for epoch in range(epochs):
    total_loss = 0.0 
    startepoch = time.time()
    for (x, y) in train_loader:
        optimizer.zero_grad()
        output = fnn(x)
        loss = L(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    endepoch = time.time()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Time: {endepoch - startepoch:.3f}s')
end = time.time()

# Eval accuracy
with torch.no_grad():
    logits = fnn(X_test_tensor)
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"Final Evaluation: Accuracy on test data: {accuracy:.3f}, Time: {end - start:.3f}")
