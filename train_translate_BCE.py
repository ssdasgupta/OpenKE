from pathlib import Path
import openke
from openke.config import Trainer, Tester
from  openke.module.model import TransE
from openke.module.model import TransIntersect, AffineBox
from openke.module.loss import MarginLoss, BCELoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import wandb
model = {
	'TransIntersect': TransIntersect,
	'AffineBox': AffineBox
}

config = {
	'dataset': 'FB15K237',
	'neg_ent': 3,
	'dim': 10,
	'score_scheme': 'conditional',
	'margin': 5.0,
	'alpha': 0.01,
	'use_gpu': False,
	'epoch': 1,
	'model': 'TransIntersect',
	'optimizer': 'adam'

}
run = wandb.init()
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
test_dataloader = TestDataLoader("./benchmarks/" + config.dataset + "/", "link")

# define the model
box_translate = model[config.model](
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = config.dim,
	p_norm = 1, 
	norm_flag = config.norm_flag,
	score_scheme = config.score_scheme)
# define the loss function
model = NegativeSampling(
	model = box_translate, 
	loss = BCELoss(),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model,
	              data_loader = train_dataloader,
	              train_times = config.epoch,
	              alpha = config.alpha,
	              use_gpu = True,
	              opt_method = config.optimizer)
trainer.run()
box_translate.save_checkpoint(Path(run.dir)/'model.ckpt')

# test the model
box_translate.load_checkpoint(Path(run.dir)/'model.ckpt')
tester = Tester(model = box_translate, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
