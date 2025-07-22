import torch

from model import DecoderOnlyModel

#import file 
DATA_PATH = None # add path to your dataset
with open (DATA_PATH, 'r', encoding='utf-8') as f:
  text = f.read()

print("first 100 characters of data: ")
print("-------------------------------------------")
print(text[:100])
print("-------------------------------------------")

# all the unique characters of data (vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocabulary: ")
print(''.join(chars))
print("\nvocabulary size: ")
print(vocab_size)

# character level tokenizer
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(f"encoding: {encode('hii there')}")
print(f"decoding: {decode(encode('hii there'))}")

# encode and convert into tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"shape of data: {data.shape}, dtype of data: {data.dtype}")

# split into training and validation data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#hyperparameters
block_size = 256
batch_size = 64
n_embd = 384
epochs = 5000
eval_interval = 100
lr = 3e-4
eval_iter = 200
dropout = 0.2
n_layer = 3
num_heads = 6


# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# loss calculation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split]= losses.mean()
    model.train()
    return out


model = DecoderOnlyModel(vocab_size, n_embd, block_size, num_heads, n_layer, dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

# training loop
for epoch in range(epochs):
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"epoch: {epoch+100} || training loss: {losses["train"]:.4f} || Validation loss: {losses["val"]:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

# generating 
context = torch.zeros((1,batch_size), dtype= torch.long)
print("Generated text")
print("-------------------------------------------")
print(decode(model.generate(context, max_new_token= 1000)[0,block_size:].tolist()))
print("-------------------------------------------")
