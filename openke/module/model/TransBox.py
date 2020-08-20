import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransIntersect(Model):

	def __init__(self,
		         ent_tot,
		         rel_tot,
		         dim = 100,
		         p_norm = 1,
		         norm_flag = False,
		         score_scheme = 'conditional', #'intersection'
		         margin = None,
		         epsilon = None):
		super(TransIntersect, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.score_scheme = score_scheme

		self.ent_embeddings1 = nn.Embedding(self.ent_tot, self.dim)
		self.ent_embeddings2 = nn.Embedding(self.ent_tot, self.dim)
		self.get_relation_embeddings()

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings1.weight.data)
			nn.init.xavier_uniform_(self.ent_embeddings2.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings1.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings2.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def get_relation_embeddings(self):
		self.rel_embeddings1 = nn.Embedding(self.rel_tot, self.dim)
		self.rel_embeddings2 = nn.Embedding(self.rel_tot, self.dim)
		if self.margin == None or self.epsilon == None:
			nn.init.xavier_uniform_(self.rel_embeddings1.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings2.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings1.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings2.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

	def _calc(self, h1, h2, t1, t2, r1, r2, mode):
		r2 = r1
		if self.norm_flag:
			h_c = (h1 + h2)/2
			t_c = (t1 + t2)/2
			r_c = (r1 + r2)/2
			h_c_ = F.normalize(h_c, 2, -1)
			t_c_ = F.normalize(t_c, 2, -1)
			r_c_ = F.normalize(r_c, 2, -1)
			delta_h = h_c_ - h_c
			delta_t = t_c_ - t_c
			delta_r = r_c_ - r_c
			h1 = h1 + delta_h
			h2 = h2 + delta_h
			t1 = t1 + delta_t
			t2 = t2 + delta_t
			r1 = r1 + delta_r
			r2 = r2 + delta_r

		if mode != 'normal':
			h1 = h1.view(-1, r1.shape[0], h1.shape[-1])
			t1 = t1.view(-1, r1.shape[0], t1.shape[-1])
			r1 = r1.view(-1, r1.shape[0], r1.shape[-1])
			h2 = h2.view(-1, r2.shape[0], h2.shape[-1])
			t2 = t2.view(-1, r2.shape[0], t2.shape[-1])
			r2 = r2.view(-1, r2.shape[0], r2.shape[-1])

		h_min = torch.min(h1, h2)
		h_max = torch.max(h1, h2)
		t_min = torch.min(t1, t2)
		t_max = torch.max(t1, t2)

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

		if self.score_scheme == 'conditional':
			score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) + torch.log(torch.nn.functional.softplus(marginal/1.0) * 1.0)
		elif self.score_scheme == 'intersection':
			score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0)
			if torch.isnan(score).any():
				import pdb; pdb.set_trace()
		elif self.score_scheme == 'boundary_distance':
			score = torch.log(torch.nn.functional.softplus(-meet/1.0) * 1.0)
		else:
			raise ValueError('Scoring scheme is not defined.')


		#score = torch.max(-meet, torch.zeros_like(meet)) # boundary distance
		#score = torch.nn.functional.softplus(-meet) # soft boundary distance
		#score = -meet # boundary distance + intersection
		#score = -torch.nn.functional.softplus(meet/1.0) * 1.0 # negative soft intersection similarity
		#score = -torch.max(meet, torch.zeros_like(meet)) # negative intersection similarity
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) # negative log soft intersection volume
		#score = torch.log(torch.nn.functional.softplus(-meet/1.0) * 1.0) # log soft boundary distance volume
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) + torch.log(torch.nn.functional.softplus(join/1.0) * 1.0)  # negative log meet-join volume ratio
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) + torch.log(torch.nn.functional.softplus(marginal/1.0) * 1.0)  # negative log conditional ratio
		#score = -torch.nn.functional.softplus(meet/1.0) * 1.0 + torch.nn.functional.softplus(join/1.0) * 1.0  # negative meet-join residual

		score = torch.sum(score, -1).flatten()

		#score = -torch.prod(torch.nn.functional.softplus(meet/1.0) * 1.0, -1).flatten() # negative soft intersection volume
		#score = -torch.prod(torch.nn.functional.softplus(meet/1.0) * 1.0 / torch.nn.functional.softplus(join/1.0) * 1.0, -1).flatten()


		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h1 = self.ent_embeddings1(batch_h)
		t1 = self.ent_embeddings1(batch_t)
		r1 = self.rel_embeddings1(batch_r)
		h2 = self.ent_embeddings2(batch_h)
		t2 = self.ent_embeddings2(batch_t)
		r2 = self.rel_embeddings2(batch_r)
		score = self._calc(h1, h2, t1, t2, r1, r2, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h1 = self.ent_embeddings1(batch_h)
		t1 = self.ent_embeddings1(batch_t)
		r1 = self.rel_embeddings1(batch_r)
		h2 = self.ent_embeddings2(batch_h)
		t2 = self.ent_embeddings2(batch_t)
		r2 = self.rel_embeddings2(batch_r)
		regul = (torch.mean(h1 ** 2) + torch.mean(h2 ** 2) + torch.mean(t1 ** 2) + torch.mean(t2 ** 2) + torch.mean(r1 ** 2) + torch.mean(r2 ** 2)) / 6
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()


class AffineBox(TransIntersect):
	"""docstring for ClassName"""
	def get_relation_embeddings(self):
		self.rel_embeddings1 = nn.Embedding(self.rel_tot, self.dim)
		self.rel_embeddings2 = nn.Embedding(self.rel_tot, self.dim)
		self.rel_embeddings1_mult = nn.Embedding(self.rel_tot, self.dim)
		self.rel_embeddings2_mult = nn.Embedding(self.rel_tot, self.dim)
		if self.margin == None or self.epsilon == None:
			nn.init.xavier_uniform_(self.rel_embeddings1.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings2.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings1_mult.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings2_mult.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings1.weight.data,
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings2.weight.data,
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings1_mult.weight.data,
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings2_mult.weight.data,
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

	def _calc(self, h1, h2, t1, t2, r1, r2, r1_mult, r2_mult, mode):
		r2 = r1
		r2_mult = r1_mult
		if self.norm_flag:
			h_c = (h1 + h2)/2
			t_c = (t1 + t2)/2
			r_c = (r1 + r2)/2
			h_c_ = F.normalize(h_c, 2, -1)
			t_c_ = F.normalize(t_c, 2, -1)
			r_c_ = F.normalize(r_c, 2, -1)
			delta_h = h_c_ - h_c
			delta_t = t_c_ - t_c
			delta_r = r_c_ - r_c
			h1 = h1 + delta_h
			h2 = h2 + delta_h
			t1 = t1 + delta_t
			t2 = t2 + delta_t
			r1 = r1 + delta_r
			r2 = r2 + delta_r

		if mode != 'normal':
			h1 = h1.view(-1, r1.shape[0], h1.shape[-1])
			t1 = t1.view(-1, r1.shape[0], t1.shape[-1])
			r1 = r1.view(-1, r1.shape[0], r1.shape[-1])
			h2 = h2.view(-1, r2.shape[0], h2.shape[-1])
			t2 = t2.view(-1, r2.shape[0], t2.shape[-1])
			r2 = r2.view(-1, r2.shape[0], r2.shape[-1])

		h_min = torch.min(h1, h2)
		h_max = torch.max(h1, h2)
		t_min = torch.min(t1, t2)
		t_max = torch.max(t1, t2)

		transfer_min = h_min * r1_mult + r1
		transfer_max = h_max * r2_mult + r2
		hr_min = torch.min(transfer_min, transfer_max)
		hr_max = torch.max(transfer_min, transfer_max)
		meet = torch.min(hr_max, t_max) - torch.max(hr_min, t_min)
		# join = torch.max(hr_max, t_max) - torch.min(hr_min, t_min)
		marginal = t_max - t_min

		if self.score_scheme == 'conditional':
			score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) + torch.log(torch.nn.functional.softplus(marginal/1.0) * 1.0)
		elif self.score_scheme == 'intersection':
			score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0)
		elif self.score_scheme == 'boundary_distance':
			score = torch.log(torch.nn.functional.softplus(-meet/1.0) * 1.0)
		else:
			raise ValueError('Scoring scheme is not defined.')


		#score = torch.max(-meet, torch.zeros_like(meet)) # boundary distance
		#score = torch.nn.functional.softplus(-meet) # soft boundary distance
		#score = -meet # boundary distance + intersection
		#score = -torch.nn.functional.softplus(meet/1.0) * 1.0 # negative soft intersection similarity
		#score = -torch.max(meet, torch.zeros_like(meet)) # negative intersection similarity
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) # negative log soft intersection volume
		#score = torch.log(torch.nn.functional.softplus(-meet/1.0) * 1.0) # log soft boundary distance volume
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) + torch.log(torch.nn.functional.softplus(join/1.0) * 1.0)  # negative log meet-join volume ratio
		#score = -torch.log(torch.nn.functional.softplus(meet/1.0) * 1.0) + torch.log(torch.nn.functional.softplus(marginal/1.0) * 1.0)  # negative log conditional ratio
		#score = -torch.nn.functional.softplus(meet/1.0) * 1.0 + torch.nn.functional.softplus(join/1.0) * 1.0  # negative meet-join residual

		score = torch.sum(score, -1).flatten()

		#score = -torch.prod(torch.nn.functional.softplus(meet/1.0) * 1.0, -1).flatten() # negative soft intersection volume
		#score = -torch.prod(torch.nn.functional.softplus(meet/1.0) * 1.0 / torch.nn.functional.softplus(join/1.0) * 1.0, -1).flatten()


		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h1 = self.ent_embeddings1(batch_h)
		t1 = self.ent_embeddings1(batch_t)
		r1 = self.rel_embeddings1(batch_r)
		r1_mult = self.rel_embeddings1_mult(batch_r)

		h2 = self.ent_embeddings2(batch_h)
		t2 = self.ent_embeddings2(batch_t)
		r2 = self.rel_embeddings2(batch_r)
		r2_mult = self.rel_embeddings2_mult(batch_r)

		score = self._calc(h1, h2, t1, t2, r1, r2, r1_mult, r2_mult, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score
