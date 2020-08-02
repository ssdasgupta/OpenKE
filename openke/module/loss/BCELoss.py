import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .Loss import Loss
_log1mexp_switch = math.log(0.5)


def log1mexp(x: torch.Tensor, split_point=_log1mexp_switch,
				exp_zero_eps=1e-7) -> torch.Tensor:
	"""
	Computes log(1 - exp(x)).

	Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).

	= log1p(-exp(x)) when x <= log(1/2)
	or
	= log(-expm1(x)) when log(1/2) < x <= 0

	For details, see

	https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

	https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
	"""
	logexpm1_switch = x > split_point
	Z = torch.zeros_like(x)
	# this clamp is necessary because expm1(log_p) will give zero when log_p=1,
	# ie. p=1
	logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
	# hack the backward pass
	# if expm1(x) gets very close to zero, then the grad log() will produce inf
	# and inf*0 = nan. Hence clip the grad so that it does not produce inf
	logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
	Z[logexpm1_switch] = logexpm1.detach() + (
	    logexpm1_bw - logexpm1_bw.detach())
	#Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
	Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

	return Z


class BCELoss(Loss):
	def __init__(self, adv_temperature = None):
		super(BCELoss, self).__init__()
		self.criterion = torch.nn.NLLLoss(reduction='mean')
		
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		log_p = torch.cat([p_score, n_score.reshape(-1, 1)])
		log1mp = log1mexp(log_p)
		logits = torch.stack([log1mp, log_p], dim=-1).reshape(-1,2)
		label = torch.cat([torch.ones_like(p_score, dtype=torch.long), 
					torch.zeros_like(n_score.reshape(-1, 1), dtype=torch.long)])
		loss = self.criterion(logits, label.flatten())
		return loss
		# if self.adv_flag:
		# 	return (self.criterion(-p_score).mean() + (self.get_weights(n_score) * self.criterion(n_score)).sum(dim = -1).mean()) / 2
		# else:
		# 	return (self.criterion(-p_score).mean() + self.criterion(n_score).mean()) / 2
			

	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()
