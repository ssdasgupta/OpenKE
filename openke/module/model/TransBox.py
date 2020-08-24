import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransBox(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransBox, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.side_length = nn.Embedding(self.ent_tot, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.side_length.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.xavier_uniform_(self.side_length.weight.data)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, h_sl, t, t_sl, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)

		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			h_sl = h_sl.view(-1, r.shape[0], h_sl.shape[-1])
			t_sl = t_sl.view(-1, r.shape[0], t_sl.shape[-1])

		h_min, h_max = torch.min(h, h + h_sl), torch.max(h, h + h_sl)
		t_min, t_max = torch.min(t, t + t_sl), torch.max(t, t + t_sl)
		r1, r2 = r, r

		if mode == 'head_batch':
			#score = h + (r - t)
			tr_min = torch.min(t_min - r1, t_max - r2)
			tr_max = torch.max(t_min - r1, t_max - r2)
			meet = torch.min(h_max, tr_max) - torch.max(h_min, tr_min)
			join = torch.max(h_max, tr_max) - torch.min(h_min, tr_min)
			marginal = h_max - h_min
		else:
			#score = (h + r) - t
			hr_min = torch.min(h_min + r1, h_max + r2)
			hr_max = torch.max(h_min + r1, h_max + r2)
			meet = torch.min(hr_max, t_max) - torch.max(hr_min, t_min)
			join = torch.max(hr_max, t_max) - torch.min(hr_min, t_min)
			marginal = hr_max - hr_min

		score = torch.nn.functional.softplus(-meet) # soft boundary distance
		#score = -torch.nn.functional.softplus(meet/1.0) * 1.0 # negative soft intersection similarity
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) # negative log soft intersection volume
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) + torch.log(torch.nn.functional.softplus(marginal/1.0) * 1.0)  # negative log conditional ratio

		score = torch.sum(score, -1).flatten()

		#score = -torch.prod(torch.nn.functional.softplus(meet/1.0) * 1.0, -1).flatten() # negative soft intersection volume
		#score = -torch.prod(torch.nn.functional.softplus(meet/1.0) * 1.0 / torch.nn.functional.softplus(join/1.0) * 1.0, -1).flatten()


		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		h_sl = self.side_length(batch_h)
		t = self.ent_embeddings(batch_t)
		t_sl = self.side_length(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h, h_sl, t, t_sl, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()
