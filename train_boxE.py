from pathlib import Path
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, BoxE, BoxEHeadTail
from openke.module.loss import BCELoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


import wandb
run = wandb.init()
config = run.config
print(config)

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/" + config.dataset + "/",
	batch_size = 100,
	threads = 8,
	sampling_mode = "normal", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = config.neg_ent,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/" + config.dataset + "/" , "link")

# define the model
BoxE_affine = BoxEHeadTail(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = config.dim, 
	init_interval_center = config.init_interval_center,
	init_interval_delta = config.init_interval_delta,
	softbox_temp = config.softbox_temp,
	)


# define the loss function
model = NegativeSampling(
	model = BoxE_affine, 
	loss = BCELoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model,
	              data_loader = train_dataloader,
	              train_times = 1000,
	              alpha = config.alpha,
	              use_gpu = True,
	              opt_method = "adam")
trainer.run()
BoxE_affine.save_checkpoint(Path(run.dir)/'model.ckpt')

# test the model
BoxE_affine.load_checkpoint(Path(run.dir)/'model.ckpt')
tester = Tester(model = BoxE_affine, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
