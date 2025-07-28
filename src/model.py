import torch, torch.nn as nn, torch.nn.functional as F

class SimpleNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, context_size=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size*embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        e = self.emb(x)
        f = e.view(e.size(0), -1)
        h = torch.tanh(self.fc1(f))
        return F.log_softmax(self.fc2(h), dim=1)

if __name__=="__main__":
    import torch
    m = SimpleNNLM(10)
    out = m(torch.randint(0,10,(4,2)))
    print("OK", out.shape)
