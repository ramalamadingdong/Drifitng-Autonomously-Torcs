from abc import abstractmethod
import datetime
from typing import Callable

import math
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models.basicnetwork import Net, SteeringNet, BrakingNet
from models.data import TrainingData
from pytocl.car import State, MPS_PER_KMH, DEGREE_PER_RADIANS


class Trainer:

    def __init__(self, extended=False):
        self.extended = extended

    def train(
        self,
        data: TrainingData,
        optimiser: torch.optim,
        loss: Callable,
        numberOfEpochs: int,
        learningRate: float,
        net = None
    ):

        if(net is None): net = self.getNetwork()
        trainingLoss = net.trainNet(data, optimiser, loss, numberOfEpochs, learningRate)
        modelName = datetime.datetime.now().strftime('%m%d%H%M%S')
        net.save(self.getModelsDir(), modelName)

        plt.plot(range(len(trainingLoss)), trainingLoss)
        # plt.show()
        plt.savefig(self.getModelsDir() + modelName + '.pdf')

    @abstractmethod
    def getNetwork(self) -> Net:
        pass

    @abstractmethod
    def getSubDirectory(self) -> str:
        pass

    def getModelsDir(self):
        return './models/models/'

    @staticmethod
    def stateToSample(state: State, extended=False) -> Variable:

        if(extended):

            # ['angle', 'speedX', 'speedY', 'speedZ', 'track0', 'track1', 'track2', 'track3', 'track4', 'track5', 'track6', 'track7', 'track8', 'track9', 'track10', 'track11', 'track12', 'track13', 'track14', 'track15', 'track16', 'track17', 'track18', 'trackPos', 'wheelSpinVel0', 'wheelSpinVel1', 'wheelSpinVel2', 'wheelSpinVel3', 'z', 'focus0', 'focus1', 'focus2', 'focus3', 'focus4']

            sample = [
                state.angle,
                state.speed_x / MPS_PER_KMH,
                state.speed_y / MPS_PER_KMH,
                state.speed_z / MPS_PER_KMH,
            ] + [
                distance
                for i, distance in enumerate(state.distances_from_edge)
            ] + [
                state.distance_from_center,
                state.wheel_velocities[0] / DEGREE_PER_RADIANS,
                state.wheel_velocities[1] / DEGREE_PER_RADIANS,
                state.wheel_velocities[2] / DEGREE_PER_RADIANS,
                state.wheel_velocities[3] / DEGREE_PER_RADIANS,
                state.z,
            ] + [
                focus for focus in state.focused_distances_from_edge
            ]

        else:
            sample = [
                 state.speed_x,
                 state.distance_from_center,
                 Trainer.degToRadians(state.angle)
            ] + [
                distance
                for i, distance in enumerate(state.distances_from_edge)
            ]

        return Variable(torch.FloatTensor(sample))

    @staticmethod
    def degToRadians(degree: float) -> float:
        return (degree * math.pi) / 180


class SteeringTrainer(Trainer):

    def getNetwork(self) -> Net:
        return SteeringNet.getPlainNetwork(self.extended)

class BrakingTrainer(Trainer):

    def getNetwork(self) -> Net:
        return BrakingNet.getPlainNetwork(self.extended)