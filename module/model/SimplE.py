import torch
import torch.nn as nn
from .Model import Model


class SimplE(Model):

    def __init__(self, ent_tot, rel_tot, dim=100):
        super(SimplE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.head_ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.tail_ent_embeddings = nn.Embedding(self.ent_tot, self.dim)

        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_inv_embeddings = nn.Embedding(self.rel_tot, self.dim)

        nn.init.xavier_uniform_(self.head_ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.tail_ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        hh = self.head_ent_embeddings(batch_h)
        ht = self.head_ent_embeddings(batch_t)

        th = self.tail_ent_embeddings(batch_h)
        tt = self.tail_ent_embeddings(batch_t)

        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)

        a = torch.sum(hh * r * tt, dim=1)
        b = torch.sum(ht * r_inv * th, dim=1)
        return torch.clamp((a + b) / 2, -20, 20)

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        hh = self.head_ent_embeddings(batch_h)
        ht = self.head_ent_embeddings(batch_t)

        th = self.tail_ent_embeddings(batch_h)
        tt = self.tail_ent_embeddings(batch_t)

        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)

        regul = (torch.mean(hh ** 2) + torch.mean(ht ** 2) + torch.mean(th ** 2) + torch.mean(tt ** 2) + torch.mean(r ** 2) + torch.mean(r_inv ** 2)) / 6
        return regul

    def predict(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        score = -self._calc_ingr(h, r, t)
        
        return score.cpu().data.numpy()
