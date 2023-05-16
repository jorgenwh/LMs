import torch

# read and prepare the articles
with open("articles.txt", "r", encoding="utf-8") as f:
    articles = []
    for line in f:
        articles.append(line)

    # add start and end tokens to each article
    articles = ['<S>' + article + '<E>' for article in articles]
    text = ''.join(articles)

# extract all the observed characters and the vocab size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode strings to int64 arrays and decode int64 arrays to strings
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[int(i)] for i in l])

# encode the whole text
data = torch.tensor(encode(text), dtype=torch.long)

# split text dataset into train and val
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size=4, block_size=8, device=None):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    if device is not None:
        x = x.to(device)
        y = y.to(device)

    return x, y

