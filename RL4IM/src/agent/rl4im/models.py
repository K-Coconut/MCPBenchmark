import random
import time

import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F

#from gensim.models import Word2Vec
from src.runner.utils import get_inv


class S2V_QN_1(torch.nn.Module):
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T, args=None):
        super(S2V_QN_1, self).__init__()
        self.args = args
        self.T = T
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        #self.mu_1 = torch.nn.Linear(1, embed_dim)
        #torch.nn.init.normal_(self.mu_1.weight,mean=0,std=0.01)
        self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))  # [#batch, #nodes, 3]
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim,True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)
        self.weight_feat = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        torch.nn.init.normal_(self.weight_feat, mean=0, std=0.01)
        self.list_pre_pooling = []
        
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)
        self.list_post_pooling = []
        
        for i in range(self.len_post_pooling):
            post_lin = torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)
        
        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)
 
    def forward(self, xv, adj, edges_weight_adj, mask=None):
        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]
        adj = edges_weight_adj

        def _mask_out(mu, mask, minibatch_size):
            # batch and type1 mode
            if minibatch_size > 1 and self.args.model_scheme == 'type1':
                mask = mask.view(minibatch_size, -1, 1)
                mask = mask.repeat(1, 1, mu.shape[-1])
                mu = mu * mask
            return mu

        mu = torch.matmul(xv, self.mu_1).clamp(0)
        mu = _mask_out(mu, mask, minibatch_size)

        mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
        mu_1 = _mask_out(mu_1, mask, minibatch_size)

        # before pooling:
        for i in range(self.len_pre_pooling):
            mu = self.list_pre_pooling[i](mu).clamp(0)

        if adj.layout == torch.sparse_coo:
            mu_pool = torch.unsqueeze(torch.sparse.mm(adj.float(), mu.squeeze()), 0)
        else:
            # sample from replay buffer
            mu_pool = torch.matmul(adj.float(), mu)
        mu_pool = _mask_out(mu_pool, mask, minibatch_size)

        # after pooling
        for i in range(self.len_post_pooling):
            mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

        mu_2 = self.mu_2(mu_pool)
        mu_2 = _mask_out(mu_2, mask, minibatch_size)

        mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(xv.transpose(1,2), mu)).view(minibatch_size, 1, -1)
        q_1 = q_1.expand(minibatch_size, nbr_node, self.embed_dim * xv.shape[-1])
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_ = q_.clamp(0)
            q = self.q(q_)

        # mask out
        if self.args.model_scheme == 'type1' and minibatch_size > 1:
            q[mask.view(minibatch_size, -1, 1)==0] = -99999
        return q


class S2V_QN_2(torch.nn.Module):
    def __init__(self,reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T, args=None):

        super(S2V_QN_2, self).__init__()
        self.T = T
        self.embed_dim=embed_dim
        self.reg_hidden=reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        #self.mu_1 = torch.nn.Linear(1, embed_dim)
        #torch.nn.init.normal_(self.mu_1.weight,mean=0,std=0.01)
        self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.mu_2, mean=0, std=0.01)
        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)
        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin =torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)
        self.q_1 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.q_1, mean=0, std=0.01)
        self.q_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.q_2, mean=0, std=0.01)
        self.q = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Parameter(torch.Tensor(2 * embed_dim, self.reg_hidden))
            torch.nn.init.normal_(self.q_reg, mean=0, std=0.01)
            self.q = torch.nn.Parameter(torch.Tensor(self.reg_hidden, 1))
        else:
            self.q = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
        torch.nn.init.normal_(self.q, mean=0, std=0.01)

    def forward(self, xv, adj, mask=None):
        #mu = self.mu_1(xv).clamp(0)
        mu = torch.matmul(xv, self.mu_1).clamp(0)
        #mu.transpose_(1,2)
        #mu_2 = self.mu_2(torch.matmul(adj, mu_init))
        #mu = torch.add(mu_1, mu_2).clamp(0)
        #mu_1 = self.mu_1(xv).clamp(0)
        mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
        #mu_1.transpose_(1,2)
        # before pooling:
        for i in range(self.len_pre_pooling):
            mu = self.list_pre_pooling[i](mu).clamp(0)

        if adj.layout == torch.sparse_coo:
            mu_pool = torch.unsqueeze(torch.sparse.mm(adj.float(), mu.squeeze()), 0)
        else:
            mu_pool = torch.matmul(adj.float(), mu)
        mu_2 = torch.matmul(mu_pool, self.mu_2) # mu_pool.shape = [1, 180, 64]

        # after pooling
        for i in range(self.len_post_pooling):
            mu_2 = self.list_post_pooling[i](mu_2).clamp(0)
        mu = torch.add(mu_1, mu_2).clamp(0) # mu_1.shape=[1, 180, 64], mu_2.shape=[1, 180, 64], mu.shape=[1, 180, 64]

        # q_1 = self.q_1(torch.matmul(xv.transpose(1,2),mu)).expand(minibatch_size,nbr_node,self.embed_dim)
        q_1 = torch.matmul(torch.matmul(xv.transpose(1,2), mu),self.q_1)    # xv.shape=[1, 180, 1], self.q_1.shape=[64, 64], q_1.shape=[1, 1, 64]
        q_1 = q_1.expand((-1, xv.shape[1], -1))
        q_2 = torch.matmul(mu,self.q_2) # self.q_2.shape=[64, 64], q_2.shape=[1, 180, 64]
        q_ = torch.cat((q_1, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = torch.matmul(q_, self.q_reg).clamp(0)
            q = torch.matmul(q_reg, self.q)
        else:
            q_=q_.clamp(0)
            q =torch.matmul(q_,self.q)
        return q


class S2V_QN(torch.nn.Module):
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):

        super(S2V_QN, self).__init__()
        self.reg_hidden = reg_hidden
        self.embed_dim = embed_dim
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Linear(1, embed_dim)
        #self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        #torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim)
        #self.mu_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        #torch.nn.init.normal_(self.mu_2, mean=0, std=0.01)

        if self.len_pre_pooling > 0:
            self.list_pre_pooling = []
            for i in range(self.len_pre_pooling):
                pre_lin = torch.nn.Linear(embed_dim, embed_dim)
                torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
                self.list_pre_pooling.append(pre_lin)

        if self.len_post_pooling > 0:
            self.list_post_pooling = []
            for i in range(self.len_post_pooling):
                pre_lin = torch.nn.Linear(embed_dim, embed_dim)
                torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
                self.list_post_pooling.append(pre_lin)

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden,bias=True)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1,bias=True)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1,bias=True)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj):

        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]


        for t in range(self.T):
            if t == 0:
                mu_1 = self.mu_1(xv)
                #mu_1 = torch.matmul(xv, self.mu_1)
                # mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                # mu = torch.add(mu_1, mu_2).clamp(0)
                mu = mu_1.clamp(0)


            else:
                mu_1 = self.mu_1(xv)
                #mu_1 = torch.matmul(xv, self.mu_1)

                # before pooling:
                if self.len_pre_pooling > 0:
                    for i in range(self.len_pre_pooling):
                        mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                if self.len_post_pooling > 0:
                    for i in range(self.len_post_pooling):
                        mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2_ = self.mu_2(mu_pool)
                #mu_2_ = torch.matmul(self.mu_2, mu_pool.transpose(1, 2))
                #mu_2_ = mu_2_.transpose(1, 2)
                mu = torch.add(mu_1, mu_2_).clamp(0)

        # q_1 = self.q_1(torch.sum( mu,dim=1).reshape(minibatch_size,1,self.embed_dim).expand(minibatch_size,nbr_node,self.embed_dim))
        xv = xv.transpose(1, 2)
        q_1 = self.q_1(torch.matmul(xv, mu))
        q_1_ = q_1.clone()
        q_1_ = q_1_.expand(minibatch_size, nbr_node, self.embed_dim)
        ####
        # mat = xv.reshape(minibatch_size, nbr_node).type(torch.ByteTensor)
        # mat = torch.ones(minibatch_size, nbr_node).type(torch.ByteTensor) - mat
        # res = torch.zeros(minibatch_size, nbr_node, nbr_node)
        # res.as_strided(mat.size(), [res.stride(0), res.size(2) + 1]).copy_(mat)
        # mu_ = mu.transpose(1, 2)
        # mu_y = torch.matmul(mu_, res)
        # mu_y = mu_y.transpose(1, 2)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1_, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_=q_.clamp(0)
            q = self.q(q_)
        return q


class W2V_QN(torch.nn.Module):
    def __init__(self, G, len_pre_pooling, len_post_pooling, embed_dim, window_size, num_paths, path_length, T):

        super(W2V_QN, self).__init__()
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Linear(1, embed_dim)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim)
        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            self.list_pre_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            self.list_post_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim)
        self.q = torch.nn.Linear(2 * embed_dim, 1)

        walks = self.build_deepwalk_corpus(G, num_paths=num_paths,
                                           path_length=path_length, alpha=0)

        self.model = Word2Vec(walks, size=embed_dim, window=window_size
                              , min_count=0, sg=1, hs=1, iter=1, negative=0, compute_loss=True)

    def random_walk(self, G, path_length, alpha=0, rand=random.Random(), start=None):

        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.nodes()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(list(nx.neighbors(G, cur))))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

    def build_deepwalk_corpus(self, G, num_paths, path_length, alpha=0):
        walks = []

        nodes = list(G.nodes())

        for cnt in range(num_paths):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(G, path_length, alpha=alpha, start=node))

        return walks

    def forward(self, xv, adj, mu_init):

        mu_w2v = torch.from_numpy(
            np.expand_dims(self.model[list(map(str, sorted(list(map(int, list(self.model.wv.vocab))))))], axis=0))

        for t in range(self.T):
            if t == 0:
                mu_1 = self.mu_1(xv)
                mu_2 = self.mu_2(torch.matmul(adj, mu_w2v))
                mu = torch.add(mu_1, mu_2).clamp(0)

            else:
                mu_1 = self.mu_1(xv)

                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(adj, mu))
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        q = self.q(q_)
        return q


class LINE_QN(torch.nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(LINE_QN, self).__init__()

        #assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.embed_dim = embed_dim
        self.order = order
        self.nodes_embeddings = torch.nn.Embedding(size, embed_dim)

        if order == 2:
            self.contextnodes_embeddings = torch.nn.Embedding(size, embed_dim)
            # Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -.5, .5) / embed_dim

        # Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):

        v_i = self.nodes_embeddings(v_i).to(device)

        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j).to(device)
            negativenodes = -self.contextnodes_embeddings(negsamples).to(device)

        else:
            v_j = self.nodes_embeddings(v_j).to(device)
            negativenodes = -self.nodes_embeddings(negsamples).to(device)

        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
        negativebatch = torch.sum(
            F.logsigmoid(
                torch.sum(mulnegativebatch, dim=2)
            ),
            dim=1)
        loss = positivebatch + negativebatch
        return -torch.mean(loss)


class BASELINE(torch.nn.Module):
    def __init__(self, size):
        super(BASELINE, self).__init__()
        self.l_1 = torch.nn.Linear(32, 3)
        self.l_2 = torch.nn.Linear(32, 32)
        self.l_3 = torch.nn.Linear(32, 1)

    def forward(self, X):
        l_1 = self.l_1(X).clamp(0)
        l_2 = self.l_2(l_1).clamp(0)
        l_3 = self.l_3(l_2)

        return l_3


class GCN_QN_1(torch.nn.Module):
    def __init__(self,reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T, args=None):

        super(GCN_QN_1, self).__init__()
        self.args = args
        self.reg_hidden = reg_hidden
        self.embed_dim = embed_dim
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.inv_adj = None


        self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)

        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim, True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)

        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)
        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)


        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        self.q = torch.nn.Linear(2 * embed_dim, 1,bias=True)
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj, mask=None):

        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]


        if adj.layout == torch.sparse_coo:
            # adj shape = [nbr_node, nbr_node]
            I = torch.sparse_coo_tensor(torch.Tensor([range(nbr_node), range(nbr_node)]), [1] * nbr_node, size=(nbr_node, nbr_node))
            adj_ = I + adj
            D = torch.sparse.sum(adj,dim=0)
            new_values = []
            for v in D.coalesce().values():
                new_values.append(max(v, 0.01))
            D = torch.sparse_coo_tensor([[0] * nbr_node, range(nbr_node)], new_values)
        else:
            I = torch.diag(torch.ones(nbr_node)).expand(minibatch_size, nbr_node, nbr_node)
            adj_ = adj + I
            D = torch.sum(adj, dim=1)
            zero_selec = np.where(D.detach().numpy() == 0)
            D[zero_selec[0], zero_selec[1]] = 0.01

        d = []
        for vec in D:
            #d.append(torch.diag(torch.rsqrt(vec)))
            if vec.layout == torch.sparse_coo:
                vec = vec.coalesce().values()
            # d.append(torch.diag(vec))
            d.append(torch.sparse_coo_tensor([range(nbr_node), range(nbr_node)], vec))
        d=torch.stack(d)

        if self.inv_adj is None or self.inv_adj.shape != adj_.shape:
            st = time.time()
            self.inv_adj = get_inv(d[0])
            ed = time.time() - st
            if self.args.mode == 'test':
                print('takes %f seconds to solve inverse matrix' % ed)
        #res = torch.zeros(minibatch_size,nbr_node,nbr_node)
        #D_=res.as_strided(res.size(), [res.stride(0), res.size(2) + 1]).copy_(D)

        #gv=torch.matmul(torch.matmul(d,adj_),d)
        if adj_.layout == torch.sparse_coo:
            gv = torch.sparse.mm(adj_, self.inv_adj)
        else:
            gv = torch.inverse(d.to_dense()).matmul(adj_)

        #mu = self.mu_1(xv).clamp(0)
        mu = torch.matmul(xv, self.mu_1).clamp(0)
        #mu.transpose_(1,2)
        #mu_2 = self.mu_2(torch.matmul(adj, mu_init))
        #mu = torch.add(mu_1, mu_2).clamp(0)

        #mu_1 = self.mu_1(xv)
        mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
        #mu_1.transpose_(1,2)
        # before pooling:
        for i in range(self.len_pre_pooling):
            mu = self.list_pre_pooling[i](mu).clamp(0)

        if gv.layout == torch.sparse_coo:
            mu_pool = torch.sparse.mm(gv, mu.squeeze())
        else:
            mu_pool = torch.matmul(gv, mu)

        for i in range(self.len_post_pooling):

            mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)



        mu_2 = self.mu_2(mu_pool)
        mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(xv.transpose(1,2),mu)).expand(minibatch_size,nbr_node,self.embed_dim)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_=q_.clamp(0)
            q = self.q(q_)
        return q

# A = to_numpy_matrix(zkc, nodelist=order)
# I = np.eye(zkc.number_of_nodes())
#
# A_hat = A + I
# D_hat = np.array(np.sum(A_hat, axis=0))[0]
# D_hat = np.matrix(np.diag(D_hat))
# def gcn_layer(A_hat, D_hat, X, W):
#     return relu(D_hat**-1 * A_hat * X * W)
#
# H_1 = gcn_layer(A_hat, D_hat, I, W_1)
# H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
#
# output = H_2
