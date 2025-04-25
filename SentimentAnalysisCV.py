import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from FNN import FNN
import torch.nn as nn
from sklearn.model_selection import KFold

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

# Set up K Fold Cross Validation
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=2)

# Metrics
accuracies = []
times = []

# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_tensor)):

    print(f'Fold {fold + 1}/{k}')
    
    # Create data loaders for this fold
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=128, sampler=val_sampler)
    
    fnn = FNN()        
    optimizer = torch.optim.AdamW(fnn.parameters(), lr=0.0001, weight_decay=0.01)
    L = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    epochs_without_improvement = 0
    patience = 3

    start = time.time()
    epochs = 10
    for epoch in range(epochs):
        # Training
        total_loss = 0.0 
        for (x, y) in train_loader:
            optimizer.zero_grad()
            output = fnn(x)
            loss = L(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation accuracy
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for (x_val, y_val) in val_loader:
                output = fnn(x_val)
                predictions = torch.argmax(output, dim=1)
                val_correct += (predictions == y_val).sum().item()
                val_total += y_val.size(0)
            val_accuracy = val_correct / val_total

        # Validation accuracy check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_without_improvement = 0
        else: epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Print loss and validation accuracy
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Validation Accuracy : {val_accuracy:.3f}')
    
    end = time.time()
    times.append(end - start) # Calculate time for each fold

    # Print time of current fold
    print(f'↑ Fold {fold + 1} — Time: {end - starttime:.2f} secs, ', end="")

    # Eval accuracy
    with torch.no_grad():
        logits = fnn(X_test_tensor)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy:.3f} ↑\n")

# Calculate average metrics
avg_accuracy = sum(accuracies) / len(accuracies)
avg_time = sum(times) / len(times)

print(f"\n\nAverage Accuracy: {avg_accuracy:.3f}")
print(f"Average Training Time per Fold: {avg_time:.2f} secs")

