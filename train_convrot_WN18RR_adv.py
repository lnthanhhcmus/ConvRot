import torch
from config import Trainer, Tester
from module.model import RotatE, NConvRot, HConvRot, ManifoldE, ConvTuckER, TransHRConv, ConvLineaRE, FuConvRot
from module.loss import SigmoidLoss
from module.strategy import NegativeSampling
from data import TrainDataLoader, TestDataLoader, ValidDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/WN18RR/",
    batch_size=1024,
    threads=8,
    sampling_mode="cross",
    bern_flag=0,
    filter_flag=1,
    neg_ent=64,
    neg_rel=0
)

valid_dataloader = ValidDataLoader("./benchmarks/WN18RR/", "link")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
rotate = HConvRot(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=512
)

# define the loss function
model = NegativeSampling(
    model=rotate,
    loss=SigmoidLoss(adv_temperature=2),
    batch_size=train_dataloader.get_batch_size(),
    regul_rate=0.0
)

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader, valid_data_loader=valid_dataloader,
                  train_times=1000, alpha=2e-5, use_gpu=True, opt_method="adam")
trainer.set_logger("WN18RR", "log/WN18RR")
trainer.run()
rotate.save_checkpoint('./checkpoint/convrot_WN18RR_adv.ckpt')

# test the model
rotate.load_checkpoint('./checkpoint/convrot_WN18RR_adv.ckpt')
tester = Tester(model=rotate, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=True)
