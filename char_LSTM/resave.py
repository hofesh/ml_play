import torch
from core import CharLSTM, save

state = torch.load("save_1", map_location="cpu")

save(state)