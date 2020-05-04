import torch
import torch.nn.functional as F

from models.Trainer import SteeringTrainer, BrakingTrainer
from models.basicnetwork import SteeringNet
from models.data_extended import ExtendedBrakingData, ExtendedSteeringData

steeringNet = SteeringNet.getPlainNetwork(extended=True)
steeringNet.load_state_dict(torch.load('models/models/steering/1126145207.model'))

SteeringTrainer(extended=True).train(
    ExtendedSteeringData(),
    torch.optim.Adam,
    F.mse_loss,
    10,
    0.00001,
    net=steeringNet
)

BrakingTrainer(extended=True).train(
    ExtendedBrakingData(),
    torch.optim.Adam,
    F.mse_loss,
    10,
    0.00001
)
