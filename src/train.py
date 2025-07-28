import os, torch, torch.nn as nn
from src.dataset import get_loaders
from src.model import SimpleNNLM

def train(corpus="data/corpus.txt", epochs=3):
    tr, vl, ds = get_loaders(corpus, context_size=2, batch_size=8)
    model = SimpleNNLM(len(ds.word2idx))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.NLLLoss()
    for e in range(1, epochs+1):
        model.train()
        lt=0
        for x,y in tr:
            opt.zero_grad()
            L=loss_fn(model(x), y)
            L.backward(); opt.step()
            lt += L.item()
        print(f"Epoch {e}/{epochs} Train loss {lt/len(tr):.4f}")
    os.makedirs("models", exist_ok=True)
    torch.save({"model_state":model.state_dict(),
                "word2idx":ds.word2idx,
                "idx2word":ds.idx2word,
                "context_size":2},
               "models/model.pt")
    print("Saved models/model.pt")

if __name__=="__main__":
    train()
