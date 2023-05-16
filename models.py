from collections import deque
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_util import *

class LSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.token_embedding_table = nn.Embedding(input_size, embedding_size)
        self.lstm_cell = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True)
        self.Who = nn.Linear(hidden_size, output_size)

    def forward(self, x, targets=None):
        #x.shape: (B, T)

        embeddings = self.token_embedding_table(x) # (B, T, embedding_size)
        logits, (h_n, c_n) = self.lstm_cell(embeddings) # (B, T, hidden_size), ..., ...
        logits = self.Who(logits) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # flatten time dimension
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens=10000):
        # x.shape: (B, T)

        # x is the context, and the context grows for each predicted token
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(x)

            # focus only on last timestep
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1) # (B, T+1)

            # remove all occurences of <S> and <E> from the s
            #s = decode(x[0])
            #s = s.replace("<S>", "")
            #s = s.replace("<E>", "")
            #print(s, end="\r")

            if decode(x[0:-3]) == "<E>":
                print("Breaking because of end sequence")
                break

        #s = decode(x[0])
        #s = s.replace("<S>", "")
        #s = s.replace("<E>", "")
        #print(s)

        return x

    def fast_generate(self, context: str, device: torch.device, max_new_tokens: int = 1000) -> None:
        # add start token to context
        context = '<S>' + context

        context = torch.tensor(encode(context), dtype=torch.long)
        context = context.reshape(1, -1) # (1, len(context))
        context = context.to(device)

        state = None
        x = context
        latest_tokens = deque(maxlen=3)

        # generate new tokens
        for _ in range(max_new_tokens):
            embeddings = self.token_embedding_table(x) # (1, T, embedding_size)
            logits, state = self.lstm_cell(embeddings, state) # (1, T, hidden_size), ..., ...
            logits = self.Who(logits) # (1, T, vocab_size)

            # focus only on last timestep
            logits = logits[:, -1, :] # becomes (1, C)

            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (1, C)

            # sample from the distribution
            x = torch.multinomial(probs, num_samples=1) # (1, 1)
            last_token = decode(x[0])
            print(f"\33[1m{last_token}\33[0m", end="", flush=True)
            latest_tokens.append(last_token)

            if "".join(latest_tokens) == "<E>":
                print("\n\33[93mBreaking because of end sequence\33[0m")
                break


