import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, BoxE
from openke.module.loss import BCELoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	batch_size = 20,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 10,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
transe = BoxE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 10, 
	margin = 6.0)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = BCELoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 2e-5, use_gpu = False, opt_method = "adam")
trainer.run()
transe.save_checkpoint('./checkpoint/transe_2.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe_2.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)
