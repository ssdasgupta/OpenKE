import torch
import torch.nn as nn
from .Model import Model

from boxes.box_wrapper import SigmoidBoxTensor, BoxTensor, DeltaBoxTensor
from boxes.modules import BoxEmbedding

class BoxE(Model):

    def __init__(self,
                ent_tot,
                rel_tot,
                dim=100,
                box_type='DeltaBoxTensor',
                init_interval_center=0.5,
                init_interval_delta=0.5,
                margin = None,
                epsilon = None):

        super(BoxE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.box_type = box_type
        breakpoint()
        self.ent_embeddings = BoxEmbedding(self.ent_tot,
                                           self.dim,
                                           self.box_type,
                               init_interval_center=init_interval_center,
                               init_interval_delta=init_interval_delta)
        self.get_relation_embeddings(self.rel_tot, self.dim)
    def get_relation_embeddings(self, num_embeddings, embedding_dim):
        self.relation_delta_weight = nn.Embedding(num_embeddings=num_embeddings,
                                                  embedding_dim=embedding_dim,
                                                  sparse=False
                                                )
        self.relation_delta_bias = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse= False)
        self.relation_min_weight = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse=False
                                                )
        self.relation_min_bias = nn.Embedding(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim,
                                              sparse= False)
        nn.init.xavier_uniform_(self.relation_delta_weight.weight.data)
        nn.init.xavier_uniform_(self.relation_delta_bias.weight.data)
        nn.init.xavier_uniform_(self.relation_min_weight.weight.data)
        nn.init.xavier_uniform_(self.relation_min_bias.weight.data)
    
    def get_relation_transform(self, box: BoxTensor, relation: torch.Tensor):
        weight_delta = self.relation_delta_weight(relation)
        weight_min = self.relation_min_weight(relation)
        bias_delta = self.relation_delta_bias(relation)
        bias_min = self.relation_min_bias(relation)
        if len(box.data.shape) == 3:
           box.data[:,0,:] = box.data[:,0,:].clone() * weight_min + bias_min
           box.data[:,1,:] = nn.functional.softplus(box.data[:,1,:].clone() * weight_delta + bias_delta)
        else:
           box.data[:,:,0,:] = box.data[:,:,0,:].clone() * weight_min + bias_min
           box.data[:,:,1,:] = nn.functional.softplus(box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return box

    def _calc(self, head, tail, relation, mode):

        if mode == 'head_batch':
            tail.data = torch.cat(head.data.shape[-3]*[tail.data])
        elif mode == 'tail_batch':
            head.data = torch.cat(tail.data.shape[-3]*[head.data])

        transformed_box = self.get_relation_transform(head, relation)
        head_tail_box_vol = transformed_box.intersection_log_soft_volume(
            tail, temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(
            temp=self.softbox_temp)

        score = torch.sum(score, -1).flatten()
        return score
    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = batch_r
        score = self._calc(h ,t, r, mode)
        return score
    
    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()