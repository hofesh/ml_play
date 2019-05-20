import gen
import torch
from core import CharLSTM

state = torch.load("save_1", map_location="cpu")

gen.save(state)