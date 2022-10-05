import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

from math import pi


class HConvRot(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0, num_filters=32, filt_h=1, filt_w=9):
        super(HConvRot, self).__init__(ent_tot, rel_tot)
        self.margin = margin
        self.epsilon = epsilon

        self.dim_e = dim * 2
        self.dim_r = dim

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

        nn.init.xavier_normal_(self.ent_embeddings.weight.data)
        nn.init.uniform_(self.rel_embeddings.weight.data, a=-pi, b=pi)

        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        # Convolution
        self.filt_h = filt_h
        self.filt_w = filt_w
        self.num_of_output_channels = num_filters

        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.dim_r*2)

        hyper_fc_length = 1*self.num_of_output_channels*self.filt_h*self.filt_w
        self.hyper_fc = torch.nn.Linear(self.dim_r, hyper_fc_length)

        fc_length = (1-self.filt_h+1)*(self.dim_r * 2 - self.filt_w+1)*self.num_of_output_channels
        self.fc = torch.nn.Linear(fc_length, self.dim_r * 2)

    def residual_convolution(self, ent_re, ent_im, rel_emb):
        ent_re = ent_re.squeeze(1)
        ent_im = ent_im.squeeze(1)
        rel_emb = rel_emb.squeeze(1)
        ent = torch.cat([ent_im, ent_re], dim=-1).view(-1, 1, 1, self.dim_r * 2)
        x = self.bn0(ent)
        x = self.inp_drop(x)
        k = self.hyper_fc(rel_emb)
        k = k.view(-1, 1, self.num_of_output_channels, self.filt_h, self.filt_w)
        k = k.view(ent.size(0)*1*self.num_of_output_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k, groups=ent.size(0))
        x = x.view(ent.size(0), 1, self.num_of_output_channels, 1-self.filt_h+1, ent.size(3)-self.filt_w+1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(ent.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return torch.chunk(x, 2, dim=1)

    def _calc(self, h, t, r, mode):
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        re_relation = torch.cos(r)
        im_relation = torch.sin(r)

        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)

        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)

        im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":

            C_3 = self.residual_convolution(re_tail, im_tail, r)
            a, b = C_3

            a = a.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
            b = b.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)

            re_score = a * re_relation * re_tail + b * im_relation * im_tail
            im_score = a * re_relation * im_tail - b * im_relation * re_tail

            re_score = re_score - re_head
            im_score = im_score - im_head

        else:

            C_3 = self.residual_convolution(re_head, im_head, r)
            a, b = C_3

            a = a.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
            b = b.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)

            re_score = a * re_head * re_relation - b * im_head * im_relation
            im_score = a * re_head * im_relation + b * im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        score = self.margin - self._calc(h, t, r, mode)
        return score

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3

        return regul
