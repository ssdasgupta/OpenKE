from pathlib import Path
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


import wandb
model = {
	'TransR': TransR
}

config = {
	'dataset': 'FB15K237',
	'neg_ent': 3,
	'dim': 10,
	'score_scheme': 'Intersection',
	'margin': 5.0,
	'alpha': 0.01,
	'use_gpu': False,
	'epoch': 1,
	'model': 'AffineBox',
	'optimizer': 'sgd',
    'norm_flag': False

}

run = wandb.init(config=config)
config = run.config
print(config)

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/" + config.dataset + "/",
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = config.neg_ent,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/" + config.dataset + "/",
	sampling_mode = 'link')

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = config.dim, 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = config.margin),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = config.dim,
	dim_r = config.dim,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = config.margin),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1, alpha = 0.5, use_gpu = config.use_gpu)
trainer.run()
parameters = transe.get_parameters()
# transe.save_parameters("./result/transr_transe.json")
transe.save_parameters(Path(run.dir)/'model.ckpt')
# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = config.epoch, alpha = config.alpha, use_gpu = config.use_gpu)
trainer.run()
transr.save_checkpoint(Path(run.dir)/'model.ckpt')

# test the model
transr.load_checkpoint(Path(run.dir)/'model.ckpt')
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = config.use_gpu)
tester.run_link_prediction(type_constrain = False)