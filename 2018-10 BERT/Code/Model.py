import torch
import torch.nn as nn

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskingModel:
    def __init__(self, tokenizer_mask_token_id: int, vocab_size: int,
                 masking_probability: float = 0.15,
                 replace_with_mask_prob: float = 0.8,
                 replace_with_random_prob: float = 0.1):
        self.tokenizer_mask_token_id = tokenizer_mask_token_id
        self.vocab_size = vocab_size
        self.masking_probability = masking_probability
        self.replace_with_mask_prob = replace_with_mask_prob
        self.replace_with_random_prob = replace_with_random_prob

    def mask(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()

        # Step 1: Sample which tokens will be masked
        probability_matrix = torch.full(input_ids.shape, self.masking_probability, device=input_ids.device)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Ignore tokens that are not masked

        # Step 2: 80% of the masked tokens -> [MASK]
        replace_with_mask = torch.bernoulli(
            torch.full(input_ids.shape, self.replace_with_mask_prob, device=input_ids.device)
        ).bool() & masked_indices
        input_ids[replace_with_mask] = self.tokenizer_mask_token_id

        # Step 3: 10% -> random token
        replace_with_random = torch.bernoulli(
            torch.full(input_ids.shape, self.replace_with_random_prob, device=input_ids.device)
        ).bool() & masked_indices & ~replace_with_mask
        random_tokens = torch.randint(self.vocab_size, input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)
        input_ids[replace_with_random] = random_tokens[replace_with_random]

        # Step 4: Remaining 10% -> unchanged (already in place)
        return input_ids, labels, masked_indices

class PositionalEncoding:
    def __init__(self, n_embd, block_size):
        self.d_model = n_embd
        self.max_seq_len = block_size
        self.pe = self._generate_positional_encodings().to(device)

    def _generate_positional_encodings(self):
        pe = torch.zeros((self.max_seq_len, self.d_model))
        position = torch.arange(0, self.max_seq_len).reshape(-1, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def get_pe(self, sequence_length):
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Requested sequence length {sequence_length} exceeds "
                f"maximum sequence length {self.max_seq_len} for positional encoding."
            )
        return self.pe[:sequence_length, :]

class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias= False) #(C, H)
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.value = nn.Linear(n_embd, head_size, bias= False)

    def forward(self, idx):
        # idx.shape = (B, T, C)
        B, T, C = idx.shape
        k = self.key(idx) # (B, T, H )
        q = self.query(idx)
        v = self.value(idx)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,T)
        wei = torch.nn.functional.softmax(wei, dim= -1)
        out = wei @ v # (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embd):
        super().__init__()
        self.head_size = n_embd // n_heads
        self.head = nn.ModuleList(Head(self.head_size, n_embd) for _ in range(n_heads))

    def forward(self, idx):
        out = torch.cat([H(idx) for H in self.head], dim= -1) # out.shape = (B,T,C)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
        self.relu = nn.ReLU()

    def forward(self, idx):
        idx = self.relu(self.linear_1(idx)) # (B, T, 4C)
        out = self.linear_2(idx) # (B, T, C)
        return out

class Block(nn.Module):
    def __init__(self, n_heads, n_embd):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, n_embd)
        self.feed_forwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, idx):
        idx = self.multi_head_attention(self.ln1(idx)) + idx # (B, T, C)
        out = self.feed_forwd(self.ln2(idx)) + idx
        return out

class EncoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, n_layers, n_embd, block_size, n_heads):
        super().__init__()
        self.masking_token = MaskingModel(vocab_size, vocab_size+1)
        self.token_embeddings_table = nn.Embedding(vocab_size+1, n_embd)
        self.positional_embeddings = PositionalEncoding(n_embd, block_size).pe
        self.blocks = nn.Sequential(*[Block(n_heads, n_embd) for _ in range(n_layers)])
        self.linear = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, Train = False):
        # idx.shape = (B, T), target.shape = (B,T)
        if Train == True:
            masked_idx, targets, masked_indices = self.masking_token.mask(idx)

            token_embd = self.token_embeddings_table(masked_idx) #idx.shape = (B, T, C)
            pos_token_embd = token_embd + self.positional_embeddings # (T,C)
            contextual_embd= self.blocks(pos_token_embd)

            logits = self.linear(contextual_embd) # (B, T, V)

            B,T,V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

            return logits, loss
        else:
            token_embd = self.token_embeddings_table(idx)
            pos_token_embd = token_embd + self.positional_embeddings
            contextual_embd = self.blocks(pos_token_embd)

            return contextual_embd