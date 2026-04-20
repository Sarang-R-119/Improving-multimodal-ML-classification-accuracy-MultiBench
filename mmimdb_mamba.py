import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MaxOut_MLP
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import MambaFusion
from training_structures.Supervised_Learning import train, test


filename = "best_mamba.pt"
traindata, validdata, testdata = get_dataloader(
    "/scratch/eecs545w26_class_root/eecs545w26_class/sarangr/mmimdb_parent/multimodal_imdb.hdf5",
    "/scratch/eecs545w26_class_root/eecs545w26_class/sarangr/mmimdb_parent/mmimdb",
    vgg=True,
    batch_size=128,
    no_robust=True,
)

encoders = [
    MaxOut_MLP(512, 512, 300, linear_layer=False).cuda(),
    MaxOut_MLP(512, 1024, 4096, 512, linear_layer=False).cuda(),
]
d_model = 256
fusion = MambaFusion(
    proj_dims=[512, 512],
    d_model=d_model,
    d_state=16,
    d_conv=2,
    expand=2,
    num_layers=1,
    use_fast_path=False,
).cuda()
head = MaxOut_MLP(23, 512, d_model, second_hidden=512).cuda()

train(
    encoders,
    fusion,
    head,
    traindata,
    validdata,
    1000,
    early_stop=True,
    task="multilabel",
    save=filename,
    optimtype=torch.optim.AdamW,
    lr=3e-4,
    weight_decay=0.01,
    objective=torch.nn.BCEWithLogitsLoss(),
    clip_val=1.0,
)

print("Testing:")
try:
    model = torch.load(filename, weights_only=False).cuda()
except TypeError:
    model = torch.load(filename).cuda()
test(
    model,
    testdata,
    method_name="mamba",
    dataset="imdb",
    criterion=torch.nn.BCEWithLogitsLoss(),
    task="multilabel",
)
