import sys
import torch

from lstm import LSTM
from data_util import vocab_size

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\33[91mError\33[0m: no context given.")
        print("Usage: python generate.py <context>")
        sys.exit(1)

    context = sys.argv[1]
    print(f"Prompting with context: '{context}'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load model
    m = LSTM(input_size=vocab_size, output_size=vocab_size)
    m.load_state_dict(torch.load('models/model0.pt'))
    m.to(device)

    with torch.no_grad():
        m.eval()
        m.fast_generate(context, device)

