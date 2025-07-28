import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class NNLMDataset(Dataset):
    def __init__(self, corpus_path, context_size=2):
        with open(corpus_path, encoding="utf-8") as f:
            text = f.read().lower().replace("\n", " ")
        tokens = text.split()
        vocab = sorted(set(tokens))
        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.idx2word = {i:w for w,i in self.word2idx.items()}
        self.data = []
        for i in range(len(tokens)-context_size):
            ctx = tokens[i:i+context_size]
            tgt = tokens[i+context_size]
            self.data.append((
                torch.tensor([self.word2idx[w] for w in ctx], dtype=torch.long),
                torch.tensor(self.word2idx[tgt], dtype=torch.long)
            ))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_loaders(corpus_path, context_size=2, batch_size=32, split_ratio=0.8):
    ds = NNLMDataset(corpus_path, context_size)
    n_train = int(len(ds)*split_ratio)
    train_ds, val_ds = random_split(ds, [n_train, len(ds)-n_train])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size),
            ds)

if __name__=="__main__":
    cl, vl, ds = get_loaders("data/corpus.txt")
    print("Vocab:", len(ds.word2idx), "Train:", len(cl.dataset), "Val:", len(vl.dataset))
