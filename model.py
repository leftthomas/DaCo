import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)

        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


class Model(nn.Module):
    def __init__(self, hidden_dim):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if not isinstance(module, (nn.Linear, nn.AdaptiveAvgPool2d)):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.g = nn.Conv2d(2048, 128, kernel_size=3)
        self.vlad = NetVLAD(num_clusters=hidden_dim // 128)

    def forward(self, x):
        feature = self.vlad(self.g(self.f(x)))
        return feature


class SimCLRLoss(nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj_1, proj_2):
        batch_size = proj_1.size(0)
        # [2*B, Dim]
        out = torch.cat([proj_1, proj_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(proj_1 * proj_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


class MoCoLoss(nn.Module):
    def __init__(self, negs, proj_dim, temperature):
        super(MoCoLoss, self).__init__()
        # init memory queue as unit random vector ---> [Negs, Dim]
        self.register_buffer('queue', F.normalize(torch.randn(negs, proj_dim), dim=-1))
        self.temperature = temperature

    def forward(self, query, key):
        batch_size = query.size(0)
        # [B, 1]
        score_pos = torch.sum(query * key, dim=-1, keepdim=True)
        # [B, Negs]
        score_neg = torch.mm(query, self.queue.t().contiguous())
        # [B, 1+Negs]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / self.temperature, torch.zeros(batch_size, dtype=torch.long, device=query.device))
        return loss

    def enqueue(self, key):
        # update queue
        self.queue.copy_(torch.cat((self.queue, key), dim=0)[key.size(0):])


class NPIDLoss(nn.Module):
    def __init__(self, n, negs, proj_dim, momentum, temperature):
        super(NPIDLoss, self).__init__()
        self.n = n
        self.negs = negs
        self.proj_dim = proj_dim
        self.momentum = momentum
        self.temperature = temperature
        # init memory bank as unit random vector ---> [N, Dim]
        self.register_buffer('bank', F.normalize(torch.randn(n, proj_dim), dim=-1))
        # z as normalizer, init with None
        self.z = None

    def forward(self, proj, pos_index):
        batch_size = proj.size(0)
        # randomly generate Negs+1 sample indexes for each batch ---> [B, Negs+1]
        idx = torch.randint(high=self.n, size=(batch_size, self.negs + 1))
        # make the first sample as positive
        idx[:, 0] = pos_index
        # select memory vectors from memory bank ---> [B, 1+Negs, Dim]
        samples = torch.index_select(self.bank, dim=0, index=idx.view(-1)).view(batch_size, -1, self.proj_dim)
        # compute cos similarity between each feature vector and memory bank ---> [B, 1+Negs]
        sim_matrix = torch.bmm(samples.to(device=proj.device), proj.unsqueeze(dim=-1)).view(batch_size, -1)
        out = torch.exp(sim_matrix / self.temperature)
        # Monte Carlo approximation, use the approximation derived from initial batches as z
        if self.z is None:
            self.z = out.detach().mean().item() * self.n
        # compute P(i|v) ---> [B, 1+Negs]
        output = out / self.z

        # compute loss
        # compute log(h(i|v))=log(P(i|v)/(P(i|v)+Negs*P_n(i))) ---> [B]
        p_d = (output.select(dim=-1, index=0) / (output.select(dim=-1, index=0) + self.negs / self.n)).log()
        # compute log(1-h(i|v'))=log(1-P(i|v')/(P(i|v')+Negs*P_n(i))) ---> [B, Negs]
        p_n = ((self.negs / self.n) / (output.narrow(dim=-1, start=1, length=self.negs) + self.negs / self.n)).log()
        # compute J_NCE(θ)=-E(P_d)-Negs*E(P_n)
        loss = - (p_d.sum() + p_n.sum()) / batch_size

        pos_samples = samples.select(dim=1, index=0)
        return loss, pos_samples

    def enqueue(self, proj, pos_index, pos_samples):
        # update memory bank ---> [B, Dim]
        pos_samples = proj.detach().cpu() * self.momentum + pos_samples * (1.0 - self.momentum)
        pos_samples = F.normalize(pos_samples, dim=-1)
        self.bank.index_copy_(0, pos_index, pos_samples)


class SimSiamLoss(nn.Module):
    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, feature_1, feature_2, proj_1, proj_2):
        sim_1 = -(proj_1 * feature_2.detach()).sum(dim=-1).mean()
        sim_2 = -(proj_2 * feature_1.detach()).sum(dim=-1).mean()
        loss = 0.5 * sim_1 + 0.5 * sim_2
        return loss


class DaCoLoss(nn.Module):
    def __init__(self, lamda, temperature):
        super(DaCoLoss, self).__init__()
        self.lamda = lamda
        self.temperature = temperature
        self.base_loss = SimCLRLoss(temperature)

    def forward(self, ori_proj_1, ori_proj_2, gen_proj_1, gen_proj_2):
        within_domain_loss = self.base_loss(ori_proj_1, ori_proj_2) + self.base_loss(gen_proj_1, gen_proj_2)
        cross_domain_loss = self.base_loss(ori_proj_1, gen_proj_1) + self.base_loss(ori_proj_1, gen_proj_2)
        loss = within_domain_loss + self.lamda * cross_domain_loss
        return loss