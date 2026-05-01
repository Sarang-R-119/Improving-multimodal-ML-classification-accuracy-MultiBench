import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test # noqa
from fusions.mult import MULTModel # noqa
from unimodals.common_models import MLP, LSTM # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import MambaFusion # noqa


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = get_dataloader('/home/vpchen/MultiBench/data/mosi_raw.pkl', robust_test=True, max_pad=True)

# Basic Perceptron
encoders = [MLP(35,20,50).cuda(),MLP(74,20,50).cuda(),MLP(300,20,50).cuda()]
# LSTM (doesn't make sense)
# encoders = [LSTM(35,40,linear_layer_outdim=50).cuda(), LSTM(74,40,linear_layer_outdim=50).cuda(), LSTM(300,40,linear_layer_outdim=50).cuda()]
fusion = MambaFusion(d_model=50, d_state=16).cuda()
# old best was 1700
head = MLP(7050,3500,1).cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=1e-3, clip_val=1.0, save='mosi_mamba_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_mamba_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=False)
