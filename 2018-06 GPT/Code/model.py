#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# self-attention
class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value= nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x.shape = (B,T,C)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim= -1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size, n_embd, dropout, block_size) for _ in range(num_head))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        out = torch.cat([H(idx) for H in self.heads], dim = -1)
        out = self.dropout(out)
        return self.proj(out)

# feed forward network
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# Transformer decoder block
class Block(nn.Module):
    def __init__(self, num_heads, n_embd, dropout, block_size):
        super().__init__()
        head_size = n_embd//num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# main class for decoder only transfomer architecture
class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embeddings = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(num_heads, n_embd, dropout, block_size) for _ in range(n_layer)])
        self.linear = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, block_size, targets = None):
        # idx and target: (B,T)
        tok_embd = self.token_embedding_table(idx) #(B,T,C)
        pos_embd = self.positional_embeddings(torch.arange(block_size)) #(T, C)
        x = tok_embd + pos_embd #(B,T,C)
        x = self.blocks(x)
        logits = self.linear(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
            return logits, loss
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits.view(B,T,C), loss
    
    def generate(self, idx, max_new_token, block_size):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx= idx_cond, block_size= block_size)
            logits = logits[:,-1, :]
            probs = F.softmax(logits, dim= -1)
            idx_next = torch.multinomial(probs, num_samples= 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx