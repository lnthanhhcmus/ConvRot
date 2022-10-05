import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

from math import pi
import numpy as np


class INConvRot(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, num_filters=32, kernel_size=3):
        super(INConvRot, self).__init__(ent_tot, rel_tot)
        self.margin = margin

        self.dim_e = dim * 2
        self.dim_r = dim

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

        nn.init.xavier_normal_(self.ent_embeddings.weight.data)
        nn.init.uniform_(self.rel_embeddings.weight.data, a=-pi, b=pi)

        self.kernel_size = kernel_size

        self.k_h = 20
        self.k_w = 10
        self.perm = 4

        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)

        self.chequer_perm = self.get_chequer_perm(dim, perm=self.perm, k_w=self.k_w, k_h=self.k_h)
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ent_tot)))
        self.register_parameter('conv_filt', torch.nn.Parameter(torch.zeros(num_filters, 1, kernel_size, kernel_size)))
        torch.nn.init.xavier_normal_(self.conv_filt)
        self.bn0 = torch.nn.BatchNorm2d(self.perm)

        # self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.in_channels_senet = num_filters*self.perm
        # self.reduction_senet = 4
        # self.senet = torch.nn.Sequential(
        #     torch.nn.Linear(self.in_channels_senet, self.in_channels_senet //
        #                     self.reduction_senet, bias=False),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(self.in_channels_senet // self.reduction_senet,
        #                     self.in_channels_senet, bias=False),
        #     torch.nn.Sigmoid()
        # )

        self.bn1 = torch.nn.BatchNorm2d(num_filters*self.perm)
        flat_sz_h = self.k_h
        flat_sz_w = 2*self.k_w
        self.bn2 = torch.nn.BatchNorm1d(2*dim)
        self.flat_sz = (flat_sz_h + self.perm) * (flat_sz_w + self.perm) * num_filters * 4 * 2
        self.fc = torch.nn.Linear(self.flat_sz, 2*dim)

    def get_chequer_perm(self, embed_dim, perm, k_w, k_h):
        ent_perm = np.int32([np.random.permutation(embed_dim) for _ in range(perm)])
        rel_perm = np.int32([np.random.permutation(embed_dim) for _ in range(perm)])

        comb_idx = []
        for k in range(perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(k_h):
                for j in range(k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx]+embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx]+embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx]+embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx]+embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).cuda()
        return chequer_perm

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def residual_convolution(self, C_1, C_2):
        # preparation
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        bs = emb_ent_real.size(0)

        emb_ent_real = emb_ent_real.squeeze(1)
        emb_ent_imag_i = emb_ent_imag_i.squeeze(1)
        emb_rel_real = emb_rel_real.squeeze(1)
        emb_rel_imag_i = emb_rel_imag_i.squeeze(1)

        real_comb_embed = torch.cat([emb_ent_real, emb_rel_real], dim=1)
        imaginary_comb_embed = torch.cat([emb_ent_imag_i, emb_rel_imag_i], dim=1)
        
        real_chequer_perm = real_comb_embed[:, self.chequer_perm]
        imaginary_chequer_perm = imaginary_comb_embed[:, self.chequer_perm]
        stacked_inp = torch.cat([real_chequer_perm, imaginary_chequer_perm], dim=-1).reshape((-1, self.perm, 2*self.k_w, self.k_h))

        x = self.bn0(stacked_inp)
        x = self.inp_drop(x)
        x = self.circular_padding_chw(x, self.kernel_size)
        x = F.conv2d(x, self.conv_filt.repeat(self.perm, 1, 1, 1), padding=0, groups=self.perm)
        # y = self.avg_pool(x).view(x.shape[0], self.in_channels_senet)
        # y = self.senet(y).view(x.shape[0], self.in_channels_senet, 1, 1)
        # x = x * y.expand_as(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(bs, self.flat_sz)
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
            C_3 = self.residual_convolution(C_1=(re_tail, im_tail), C_2=(re_relation, im_relation))
            a, b = C_3

            a = a.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
            b = b.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)

            re_score = a * re_relation * re_tail + b * im_relation * im_tail
            im_score = a * re_relation * im_tail - b * im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head

        else:
            C_3 = self.residual_convolution(C_1=(re_head, im_head), C_2=(re_relation, im_relation))
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

        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul
