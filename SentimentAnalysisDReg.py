import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from FNN import FNN
import torch.nn as nn
from collections import Counter

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
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=512)

# Create FNN model
fnn_dropout = FNN(use_dropout=True, dropout_rate=0.5)
fnn_dropout.train()
optimizer = torch.optim.AdamW(fnn.parameters(), lr=0.0001, weight_decay=0.01)
L = nn.CrossEntropyLoss()

# Training Dropout Regularization Loop
epochs = 6
print("Training of dropout regularization FNN model started...")
start = time.time()
for epoch in range(epochs):
    total_loss = 0.0 
    startepoch = time.time()
    for (x, y) in train_loader:
        optimizer.zero_grad()
        output = fnn_dropout(x)
        loss = L(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    endepoch = time.time()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Epoch Time: {endepoch - startepoch:.3f}s')
end = time.time()


# Train a baseline model without dropout
fnn_baseline = FNN(use_dropout=False)
fnn_baseline.train()
optimizer_baseline = torch.optim.AdamW(fnn_baseline.parameters(), lr=0.0001, weight_decay=0.01)
L_2 = nn.CrossEntropyLoss()

# Training Baseline Loop
print("Training of baseline FNN model started...")
start_baseline = time.time()
for epoch in range(epochs):
    total_loss = 0.0 
    startepoch = time.time()
    for (x, y) in train_loader:
        optimizer_baseline.zero_grad()
        output = fnn_baseline(x)
        loss = L_2(output, y)
        loss.backward()
        optimizer_baseline.step()
        total_loss += loss.item()
    endepoch = time.time()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Epoch Time: {endepoch - startepoch:.3f}s')
end_baseline = time.time()
print("Training Completed\n")


# Evaluate the accuracy of baseline and dropout models
fnn_dropout.eval()
fnn_baseline.eval()
with torch.no_grad():
    # baseline
    logits_baseline     = fnn_baseline(X_test_tensor)
    predictions_baseline = torch.argmax(logits_baseline, dim=1)
    accuracy_baseline    = (predictions_baseline == y_test_tensor).sum().item() / len(y_test_tensor)
    # dropout
    logits_dropout      = fnn_dropout(X_test_tensor)
    predictions_dropout  = torch.argmax(logits_dropout, dim=1)
    accuracy_dropout     = (predictions_dropout == y_test_tensor).sum().item() / len(y_test_tensor)
print(f"\nBaseline Evaluation - Accuracy on test data: {accuracy_baseline:.3f}")
print(f"Dropout Evaluation - Accuracy on test data: {accuracy_dropout:.3f}")

# 5.2 Train 5 different dropout models using bagging.
ensemble_models = []
ensemble_times = []
print("\nTraining bagged ensemble models…")
for i in range(5):
    # 1) bootstrap sample indices
    n = len(X_train_tensor)
    idxs = torch.randint(0, n, (n,))
    sampler = SubsetRandomSampler(idxs)
    bag_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=128,
        sampler=sampler
    )

    # train one dropout model on this
    m = FNN(use_dropout=True, dropout_rate=0.5)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=0.0001, weight_decay=0.01)
    start_bag = time.time()
    for epoch in range(epochs):
        for xb, yb in bag_loader:
            opt.zero_grad()
            loss = L(m(xb), yb)
            loss.backward()
            opt.step()
    time_bag = time.time() - start_bag

    ensemble_models.append(m)
    ensemble_times.append(time_bag)
    print(f"  • Trained model {i+1}/5")

# Ensemble evaluation by majority vote
print("\nEvaluating bagged ensemble…")
m.eval()
with torch.no_grad():
    # get the predictions
    all_preds = torch.stack([
        model(X_test_tensor).argmax(dim=1)
        for model in ensemble_models
    ])  # shape (5, N_test)

    votes = []
    for col in all_preds.t():
        vote_counts = Counter(col.tolist())
        votes.append(vote_counts.most_common(1)[0][0])
    ensemble_preds = torch.tensor(votes)

    accuracy_ensemble = (ensemble_preds == y_test_tensor).sum().item() / len(y_test_tensor)

# Compare bagging to baseline
print(f"\nBagging Ensemble Evaluation - Accuracy on test data: {accuracy_ensemble:.3f}")
print(f"\nBaseline Evaluation           - Accuracy on test data: {accuracy_baseline:.3f}")

# Compare time costs
baseline_time = end_baseline - start_baseline
dropout_time = end - start

print(f"Baseline model training time   : {baseline_time:.2f}s")
print(f"Dropout model training time    : {dropout_time:.2f}s")
for idx, t in enumerate(ensemble_times, 1):
    print(f"Bagged model {idx} training time: {t:.2f}s")