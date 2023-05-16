from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_util import get_batch, vocab_size, decode
from models import LSTM

#torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
BLOCK_SIZE = 512
LEARNING_RATE = 1e-3
TRAIN_ITERS = 1000


if __name__ == "__main__":
    m = LSTM(input_size=vocab_size, embedding_size=1024, hidden_size=1024, output_size=vocab_size)
    m = m.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)
    losses = deque(maxlen=1000)

    for i in range(TRAIN_ITERS):
        xb, yb = get_batch("train", BATCH_SIZE, BLOCK_SIZE, device)

        logits, loss = m(xb, targets=yb)
        losses.append(loss.item())

        if i % 100 == 0:
            print(f"Iteration {i}: {sum(losses) / len(losses)}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0 or i == TRAIN_ITERS - 1:
            torch.save(m.state_dict(), "models/model{i}.pt")
