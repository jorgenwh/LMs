from collections import deque
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_util import *

class LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.embedding_size = 1024
        self.lstm_hidden_size = 1024

        self.token_embedding_table = nn.Embedding(input_size, self.embedding_size)
        self.lstm_cell = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=2,
                batch_first=True)
        self.Who1 = nn.Linear(self.lstm_hidden_size, 512)
        self.Who2 = nn.Linear(512, output_size)

    def forward(self, x, targets=None):
        #x.shape: (B, T)

        embeddings = self.token_embedding_table(x) # (B, T, embedding_size)
        logits, (h_n, c_n) = self.lstm_cell(embeddings) # (B, T, lstm_hidden_size), ..., ...
        logits = self.Who1(logits) # (B, T, hidden_size)
        logits = self.Who2(logits) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # flatten time dimension
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def fast_generate(self, context: str, device: torch.device, max_new_tokens: int = 1000) -> None:
        # add start token to context
        context = '<S>' + context

        context = torch.tensor(encode(context), dtype=torch.long)
        context = context.reshape(1, -1) # (1, len(context))
        context = context.to(device)

        state = None
        x = context
        latest_tokens = deque(maxlen=3)

        print("<S>")
        # generate new tokens
        for _ in range(max_new_tokens):
            embeddings = self.token_embedding_table(x) # (1, T, embedding_size)
            logits, state = self.lstm_cell(embeddings, state) # (1, T, lstm_hidden_size), ..., ...
            logits = self.Who1(logits) # (1, T, hidden_size)
            logits = self.Who2(logits) # (1, T, vocab_size)

            # focus only on last timestep
            logits = logits[:, -1, :] # becomes (1, C)

            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (1, C)

            # sample from the distribution using 
            # multinomial for probabilistic sampling or argmax for greedy sampling
            x = torch.multinomial(probs, num_samples=1) # (1, 1)
            #x = torch.argmax(probs, dim=-1).reshape(1, 1)

            last_token = decode(x[0])
            latest_tokens.append(last_token)

            print(f"{last_token}", end="", flush=True)

            if "".join(latest_tokens) == "<E>":
                print()
                break

