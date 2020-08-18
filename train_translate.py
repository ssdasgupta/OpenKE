from pathlib import Path
import openke
from openke.config import Trainer, Tester
from  openke.module.model import TransE
from openke.module.model import TransIntersect
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import wandb
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
box_translate = TransIntersect(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = config.dim,
	p_norm = 1, 
	norm_flag = True)
# define the loss function
model = NegativeSampling(
	model = box_translate, 
	loss = MarginLoss(margin = config.margin),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model,
	              data_loader = train_dataloader,
	              train_times = 1000,
	              alpha = config.alpha,
	              use_gpu = True)
trainer.run()
box_translate.save_checkpoint(Path(run.dir)/'model.ckpt')

# test the model
box_translate.load_checkpoint(Path(run.dir)/'model.ckpt')
tester = Tester(model = box_translate, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
