import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

from math import pi

class NConvRot(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0, num_filters=32, kernel_size=3):
        super(NConvRot, self).__init__(ent_tot, rel_tot)
        self.margin = margin
        self.epsilon = epsilon

        self.dim_e = dim * 2
        self.dim_r = dim

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

        nn.init.xavier_normal_(self.ent_embeddings.weight.data)
        nn.init.uniform_(self.rel_embeddings.weight.data, a=-pi, b=pi)

        self.kernel_size = kernel_size
        self.num_of_output_channels = num_filters
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = self.dim_r * 4 * self.num_of_output_channels

        self.fc = torch.nn.Linear(self.fc_num_input, self.dim_r * 2)

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)

        self.bn_conv2 = torch.nn.BatchNorm1d(self.dim_r * 2)

        self.feature_map_dropout = torch.nn.Dropout2d(0.2)

        nn.init.xavier_normal_(self.conv1.weight.data)
        nn.init.xavier_normal_(self.fc.weight.data)

    def residual_convolution(self, C_1, C_2):

        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2

        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.dim_r),
                       emb_ent_imag_i.view(-1, 1, 1, self.dim_r),
                       emb_rel_real.view(-1, 1, 1, self.dim_r),
                       emb_rel_imag_i.view(-1, 1, 1, self.dim_r)], 2)

        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn_conv2(self.fc(x)))
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

            re_score = re_relation * a + im_relation * b
            im_score = re_relation * a - im_relation * b
            re_score = re_score - re_head
            im_score = im_score - im_head

        else:
            C_3 = self.residual_convolution(C_1=(re_head, im_head), C_2=(re_relation, im_relation))
            a, b = C_3

            a = a.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
            b = b.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)

            re_score = a * re_relation - b * im_relation
            im_score = a * im_relation + b * re_relation
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
