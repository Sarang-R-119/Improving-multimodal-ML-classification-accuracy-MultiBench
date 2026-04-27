from datasets.mimic.get_data import get_dataloader # noqa
from fusions.common_fusions import IMDBMambaFusion, MimicMambaFusion # noqa
from training_structures.Supervised_Learning import train, test # noqa
from unimodals.common_models import MLP, GRU # noqa

import sys
import os
import torch
from torch import nn

sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
# traindata, validdata, testdata = get_dataloader(
#     7, imputed_path='/home/pliang/yiwei/im.pk')
traindata, validdata, testdata = get_dataloader(
    7, imputed_path='/scratch/eecs545w26_class_root/eecs545w26_class/sarangr/mimic/im.pk')


encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRU(
    12, 30, dropout=True, batch_first=True).cuda()]
# fusion = IMDBMambaFusion(proj_dims=[10, 30], d_model=128).cuda()
fusion = MimicMambaFusion(d_model=32, d_static=10, d_time=30).cuda()
head = MLP(64, 40, 2, dropout=False).cuda()

# train
train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

# test
print("Testing: ")
model = torch.load('best.pt', weights_only=False).cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 7', auprc=True)
