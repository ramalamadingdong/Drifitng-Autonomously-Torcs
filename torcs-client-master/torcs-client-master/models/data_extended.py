from abc import abstractmethod

import sqlite3

import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from models.data import TrainingData


class ExtendedData(TrainingData):

    def __init__(self, track='alpine_1'):

        db = sqlite3.connect('training-data/trainingData.db')

        sqlData = 'SELECT ' \
                  + list2list(self.getDataColumns()) \
                  + ' FROM observations'
                  # + ' FROM observations WHERE track = \'' + track + '\''

        sqlTarget = 'SELECT ' \
                  + list2list(self.getTargetColumns()) \
                  + ' FROM observations'
                  # + ' FROM observations WHERE track = \'' + track + '\''

        self.data = DataFrame(db.execute(
            sqlData
        ).fetchall())

        self.targets = DataFrame(db.execute(
            sqlTarget
        ).fetchall())

        db.close()

    # def __getitem__(self, index):
    #     return torch.FloatTensor(list(self.data.loc[index, :].values)), torch.FloatTensor(list(self.targets.loc[index, :].values))

    # def __len__(self):
    #     return len(self.data)

    # @abstractmethod
    # def getDataColumns(self) -> []:
    #     pass
    #
    # @abstractmethod
    # def getTargetColumns(self) -> []:
    #     pass

class ExtendedBrakingData(ExtendedData):

    def getDataColumns(self) -> []:
        return ['angle', 'speedX', 'speedY', 'speedZ', 'track0', 'track1', 'track2', 'track3', 'track4', 'track5', 'track6', 'track7', 'track8', 'track9', 'track10', 'track11', 'track12', 'track13', 'track14', 'track15', 'track16', 'track17', 'track18', 'trackPos', 'wheelSpinVel0', 'wheelSpinVel1', 'wheelSpinVel2', 'wheelSpinVel3', 'z', 'focus0', 'focus1', 'focus2', 'focus3', 'focus4']

    def getTargetColumns(self) -> []:
        return ['brake']

def list2list(list: []) -> str:

    string = ''

    for entry in list:
        string += entry + ', '

    return string[:-2]

class ExtendedSteeringData(ExtendedData):

    def getDataColumns(self) -> []:
        return ['angle', 'speedX', 'speedY', 'speedZ', 'track0', 'track1', 'track2', 'track3', 'track4', 'track5', 'track6', 'track7', 'track8', 'track9', 'track10', 'track11', 'track12', 'track13', 'track14', 'track15', 'track16', 'track17', 'track18', 'trackPos', 'wheelSpinVel0', 'wheelSpinVel1', 'wheelSpinVel2', 'wheelSpinVel3', 'z', 'focus0', 'focus1', 'focus2', 'focus3', 'focus4']

    def getTargetColumns(self) -> []:
        return ['steer']
