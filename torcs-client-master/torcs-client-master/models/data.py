from abc import abstractmethod

import torch
import math

from pandas import DataFrame
from torch.utils.data import Dataset


class TrainingData(Dataset):

    @abstractmethod
    def getTargetColumns(self) -> list:
        pass

    def transformData(self, dataframe: DataFrame):
        return dataframe

    def getDataColumns(self) -> list:
        return ['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS'] + ['TRACK_EDGE_' + str(i) for i in range(19)]

    def __init__(self, dataframe: DataFrame = None):
        self.targets = DataFrame()
        self.data = DataFrame()

        if(dataframe is not None): self.append(dataframe)

    def append(self, dataframe: DataFrame):
        dataframe = self.transformData(dataframe)
        dataframe['ANGLE_TO_TRACK_AXIS'] = dataframe['ANGLE_TO_TRACK_AXIS'] * math.pi / 180

        dataframe.index = range(self.__len__(), self.__len__() + len(dataframe))
        self.targets = self.targets.append(dataframe.loc[:, self.getTargetColumns()])
        self.data = self.data.append(dataframe.loc[:, self.getDataColumns()])

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, item: int) -> (torch.FloatTensor, torch.FloatTensor):
        return torch.FloatTensor(list(self.data.loc[item, :].values)), torch.FloatTensor(list(self.targets.loc[item, :].values))


class SteeringTrainingData(TrainingData):

    def getTargetColumns(self) -> list:
        return ['STEERING']

    def transformData(self, dataframe: DataFrame):
        dataframe['STEERING'] = (dataframe['STEERING'] + 1) / 2
        return dataframe


class BrakingTrainingData(TrainingData):

    def getTargetColumns(self) -> list:
        return ['BRAKE']

    def transformData(self, dataframe: DataFrame):
        dataframe['BRAKE'] = round(dataframe['BRAKE'], 0)
        return dataframe