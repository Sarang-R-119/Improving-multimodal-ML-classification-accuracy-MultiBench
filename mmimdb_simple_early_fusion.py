import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MaxOut_MLP, Identity
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test


filename = "best_ef.pt"
# traindata, validdata, testdata = get_dataloader(
#     "../video/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=128)
traindata, validdata, testdata = get_dataloader(
    "/scratch/eecs545w26_class_root/eecs545w26_class/sarangr/mmimdb_parent/multimodal_imdb.hdf5",
    "/scratch/eecs545w26_class_root/eecs545w26_class/sarangr/mmimdb_parent/mmimdb",
    # vgg=True: use 4096-dim features stored in the HDF5 (no live VGG). Matches head input 300 + 4096 = 4396.
    vgg=True,
    batch_size=128,
    no_robust=True,
)

encoders = [Identity(), Identity()]
head = MaxOut_MLP(23, 512, 4396).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 1000,early_stop=True, task="multilabel",
      save=filename, optimtype=torch.optim.AdamW, lr=4e-2, weight_decay=0.01, objective=torch.nn.BCEWithLogitsLoss())

print("Testing:")
# PyTorch 2.6+: torch.load defaults to weights_only=True; full checkpoints need False.
try:
    model = torch.load(filename, weights_only=False).cuda()
except TypeError:
    model = torch.load(filename).cuda()
test(model, testdata, method_name="ef", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
