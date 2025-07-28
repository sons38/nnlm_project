import torch
from src.model import SimpleNNLM

class NNLMInfer:
    def __init__(self, path="models/model.pt"):
        ck = torch.load(path, map_location="cpu")
        self.w2i,self.i2w = ck["word2idx"], ck["idx2word"]
        self.cs = ck["context_size"]
        self.m = SimpleNNLM(len(self.w2i), context_size=self.cs)
        self.m.load_state_dict(ck["model_state"]); self.m.eval()
    def predict(self, ctx):
        idxs=[self.w2i[w] if w in self.w2i else 0 for w in ctx]
        x=torch.tensor([idxs])
        lp = self.m(x).squeeze(0)
        return self.i2w[lp.argmax().item()]

if __name__=="__main__":
    inf=NNLMInfer()
    print("Next:", inf.predict(["안녕하세요","저는"]))
