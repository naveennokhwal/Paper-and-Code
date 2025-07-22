import torch
import tiktoken

from Model import EncoderOnlyModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
DATASET_PATH = None # Add path to your dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# tokenization
enc = tiktoken.get_encoding("p50k_base")
data = enc.encode(text)
print(data[:100])

# vocabulary
vocab_size = enc.n_vocab
print(f"vocab size: {vocab_size}")

# training and validation data
n = int(0.9 * len(data))
train_data = torch.tensor(data[:n]).to(device)
val_data = torch.tensor(data[n:]).to(device)
print(f"train data: {train_data.shape, train_data.dtype} || val data: {val_data.shape, val_data.dtype}")

# hyperparameters
batch_size = 4
block_size = 8
n_embd = 64
n_heads = 2
n_layers = 2
eval_iter= 200
epochs = 3000
eval_interval = 500
lr = 3e-4

# sample generator
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    return x

# loss calculation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter, device=device)
        for k in range(eval_iter):
            X = get_batch(split)
            _, loss = model(X, Train=True)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create model and move it to device
model = EncoderOnlyModel(vocab_size, n_layers, n_embd, block_size, n_heads).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# training loop
for epoch in range(epochs):
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"epoch: {epoch+100} || training loss: {losses['train']:.4f} || Validation loss: {losses['val']:.4f}")
    xb = get_batch("train")
    _, loss = model(xb, Train=True)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()