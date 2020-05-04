from abc import abstractmethod
from typing import Callable
import json

import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models.data import SteeringTrainingData, TrainingData


class Net(torch.nn.Module):

    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int):
        super(Net, self).__init__()

        self.h1 = torch.nn.Linear(inputSize, hiddenSize)
        self.h2 = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, x: Variable) -> Variable:

        x = self.h1(x)
        x = F.sigmoid(x)
        x = self.h2(x)
        x = F.sigmoid(x)

        return x

    def trainNet(self,
         data: TrainingData,
         optimiserFunction: Callable = torch.optim.SGD,
         lossFunction: Callable = F.mse_loss,
         numberOfEpochs: int = 2,
         learningRate: float = 0.001
    ) -> list:

        optimiser = optimiserFunction(self.parameters(), lr=learningRate)

        dataLoader = DataLoader(data, shuffle=True)

        losses = []
        modelName = datetime.datetime.now().strftime('%m%d%H%M%S')

        for epoch in range(numberOfEpochs):

            print('Epoch ' + str(epoch))
            totalLoss = 0

            for i, sample in enumerate(dataLoader):
                inputs, targets = sample

                inputs = Variable(inputs)
                targets = Variable(targets)

                optimiser.zero_grad()

                outputs = self(inputs)
                loss = lossFunction(outputs, targets)
                totalLoss += loss
                loss.backward()
                optimiser.step()

            losses.append(totalLoss.data.numpy()[0])


            self.save('models/models/', modelName)
            plt.plot(range(len(losses)), losses)
            plt.savefig('models/models/steering/' + modelName + '.pdf')

        return losses

    def save(self, directory: str, modelName: str):
        torch.save(self.state_dict(), directory + self.subdirectory() + modelName + '.model')

    def predictionToFloat(self, prediction: Variable) -> float:
        return prediction.data.numpy()[0]

    @abstractmethod
    def subdirectory(self) -> str:
        pass

    @abstractmethod
    def predict(self, sample: Variable):
        pass

    @staticmethod
    @abstractmethod
    def getPlainNetwork():
        pass

class SteeringNet(Net):

    def subdirectory(self) -> str:
        return 'steering/'

    def predict(self, sample: Variable) -> float:
        prediction = self.predictionToFloat(self(sample))
        # return prediction * 2 - 1
        return prediction

    @staticmethod
    def getPlainNetwork(extended=True):
        if (extended): return SteeringNet(34, 60, 1)
        else: return SteeringNet(22, 60, 1)

    def forward(self, x: Variable) -> Variable:

        x = self.h1(x)
        x = F.sigmoid(x)
        x = self.h2(x)

        return x


class BrakingNet(Net):

    def subdirectory(self) -> str:
        return 'braking/'

    def predict(self, sample: Variable) -> float:
        prediction = round(self.predictionToFloat(self(sample)))
        return prediction

    @staticmethod
    def getPlainNetwork(extended=True):
        if (extended): return BrakingNet(34, 60, 1)
        else: return BrakingNet(22, 60, 1)

    def forward(self, x: Variable) -> Variable:

        x = self.h1(x)
        x = F.sigmoid(x)
        x = self.h2(x)
        x = F.sigmoid(x)

        return x

class Meta():

    def __init__(self):
        self.data = {}
        self.loaded = False

    @staticmethod
    def load(file: str = './models/models/00_meta.json'):
        meta = Meta()
        meta.data = json.load(open(file))
        meta.loaded = True

        return meta

    def append(self, model: str, data: {}):

        if(model in self.data):
            for key, value in data:
                self.data[model][key] = value

        else:
            self.data[model] = data

    def save(self, file: str = './models/models/00_meta.json'):

        if(not self.loaded): raise Exception('Must load meta file first')

        with open(file, 'w') as metaFile:
            json.dump(self.data, metaFile)