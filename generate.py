import sys
import torch

from models import LSTM
from data_util import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\33[91mError\33[0m: no context given.")
        print("Usage: python generate.py <context>")
        sys.exit(1)

    context = sys.argv[1]
    print(f"Prompting with context: '{context}'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # add start token to context
    #context = '<S>' + context

    #context = torch.tensor(encode(context), dtype=torch.long)
    #context = context.reshape(1, -1)
    #context = context.to(device)

    # load model
    m = LSTM(input_size=vocab_size, embedding_size=1024, hidden_size=1024, output_size=vocab_size)
    m.load_state_dict(torch.load('models/model0.pt'))
    m.to(device)

    with torch.no_grad():
        m.eval()
        m.fast_generate(context, device)

