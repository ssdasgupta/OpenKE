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
                softbox_temp=10,
                margin = None,
                epsilon = None):

        super(BoxE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.box_type = box_type
        self.softbox_temp = softbox_temp
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
            ratio = int(head.data.shape[-3] / tail.data.shape[-3])
            tail.data = torch.cat(ratio * [tail.data])
            relation = torch.cat(ratio * [relation])
        elif mode == 'tail_batch':
            ratio = int(tail.data.shape[-3] / head.data.shape[-3])
            head.data = torch.cat(ratio * [head.data])
            relation = torch.cat(ratio * [relation])
        transformed_box = self.get_relation_transform(head, relation)
        head_tail_box_vol = transformed_box.intersection_log_soft_volume(
            tail, temp=self.softbox_temp)
        score = head_tail_box_vol - tail.log_soft_volume(
           temp=self.softbox_temp)
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


class BoxEHeadTail(BoxE):
    """docstring for BoxEHeadTail"""
    def __init__(self,
                ent_tot,
                rel_tot,
                dim=100,
                box_type='DeltaBoxTensor',
                init_interval_center=0.5,
                init_interval_delta=0.5,
                softbox_temp=10,
                margin = None,
                epsilon = None):
        super(BoxEHeadTail, self).__init__(ent_tot,
                rel_tot,
                dim=dim,
                box_type=box_type,
                init_interval_center=init_interval_center,
                init_interval_delta=init_interval_delta,
                softbox_temp=softbox_temp,
                margin = None,
                epsilon = None)

    def get_relation_embeddings(self, num_embeddings: int, embedding_dim: int):
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

        self.relation_delta_weight_tail = nn.Embedding(num_embeddings=num_embeddings,
                                                  embedding_dim=embedding_dim,
                                                  sparse=False
                                                )
        self.relation_delta_bias_tail = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse= False)
        self.relation_min_weight_tail = nn.Embedding(num_embeddings=num_embeddings,
                                                embedding_dim=embedding_dim,
                                                sparse=False
                                                )
        self.relation_min_bias_tail = nn.Embedding(num_embeddings=num_embeddings,
                                              embedding_dim=embedding_dim,
                                              sparse= False)
        nn.init.xavier_uniform_(self.relation_delta_weight_tail.weight.data)
        nn.init.xavier_uniform_(self.relation_delta_bias_tail.weight.data)
        nn.init.xavier_uniform_(self.relation_min_weight_tail.weight.data)
        nn.init.xavier_uniform_(self.relation_min_bias_tail.weight.data)

    def get_relation_transform_tail(self, box: BoxTensor, relation: torch.Tensor):
        weight_delta = self.relation_delta_weight_tail(relation)
        weight_min = self.relation_min_weight_tail(relation)
        bias_delta = self.relation_delta_bias_tail(relation)
        bias_min = self.relation_min_bias_tail(relation)
        if len(box.data.shape) == 3:
           box.data[:,0,:] = box.data[:,0,:].clone() * weight_min + bias_min
           box.data[:,1,:] = nn.functional.softplus(box.data[:,1,:].clone() * weight_delta + bias_delta)
        else:
           box.data[:,:,0,:] = box.data[:,:,0,:].clone() * weight_min + bias_min
           box.data[:,:,1,:] = nn.functional.softplus(box.data[:,:,1,:].clone() * weight_delta + bias_delta)
        return box

    def _calc(self, head, tail, relation, mode):
        if mode == 'head_batch':
            ratio = int(head.data.shape[-3] / tail.data.shape[-3])
            tail.data = torch.cat(ratio * [tail.data])
            relation = torch.cat(ratio * [relation])
        elif mode == 'tail_batch':
            ratio = int(tail.data.shape[-3] / head.data.shape[-3])
            head.data = torch.cat(ratio * [head.data])
            relation = torch.cat(ratio * [relation])
        
        transformed_box = self.get_relation_transform(head, relation)
        transformed_box_tail = self.get_relation_transform_tail(tail, relation)
        head_tail_box_vol = transformed_box.intersection_log_soft_volume(
            transformed_box_tail, temp=self.softbox_temp)
        score = head_tail_box_vol - transformed_box_tail.log_soft_volume(
           temp=self.softbox_temp)
        return score
