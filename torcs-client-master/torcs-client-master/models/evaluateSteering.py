import pandas
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models.Trainer import Trainer
from models.data import SteeringTrainingData, BrakingTrainingData
from models.data_extended import ExtendedSteeringData
from my_driver import MyDriver
from pytocl.car import State, MPS_PER_KMH, DEGREE_PER_RADIANS


def dataToState(data) -> State:
    datalist = list(data)
    state = State({
        'angle': datalist[2] / DEGREE_PER_RADIANS,
        'curLapTime': 0,
        'damage': 0,
        'distFromStart': 0,
        'distRaced': 0,
        'fuel': 0,
        'gear': 0,
        'lastLapTime': 0,
        'opponents': {},
        'racePos': 0,
        'rpm': 0,
        'speedX': datalist[0] / MPS_PER_KMH,
        'speedY': 0,
        'speedZ': 0,
        'track': datalist[3:22],
        'trackPos': datalist[1],
        'wheelSpinVel': (0, 0),
        'z': 0,
        'focus': (0,0)
    })
    return state

def dataToStateExtended(data) -> State:
    datalist = list(data)
    state = State({
        'angle': datalist[0],
        'curLapTime': 0,
        'damage': 0,
        'distFromStart': 0,
        'distRaced': 0,
        'fuel': 0,
        'gear': 0,
        'lastLapTime': 0,
        'opponents': {},
        'racePos': 0,
        'rpm': 0,
        'speedX': datalist[1],
        'speedY': datalist[2],
        'speedZ': datalist[3],
        'track': datalist[4:23],
        'trackPos': datalist[23],
        'wheelSpinVel': (
            datalist[24], datalist[25], datalist[26], datalist[27],
        ),
        'z': datalist[28],
        'focus': (
            datalist[29], datalist[30], datalist[31], datalist[32], datalist[33]
        )
    })
    return state


def predictionToFloat(prediction: Variable) -> float:
    return prediction.data.numpy()[0]

# steeringData = SteeringTrainingData(pandas.read_csv('training-data/train_data/alpine-1.csv'))
# brakingData = BrakingTrainingData(pandas.read_csv('training-data/train_data/alpine-1.csv'))

steeringData = ExtendedSteeringData()

observations = []
driver = MyDriver()

for i in range(len(steeringData)):

    state = dataToStateExtended(steeringData[i][0])
    sample = Trainer.stateToSample(state)

    command = driver.drive(state)
    #
    # observation = {
    #     'target': (brakingData[i][1]).numpy()[0],
    #     'actual': command.brake
    # }

    observation = {
        'target': (steeringData[i][1]).numpy()[0],
        'actual': command.steering
    }

    observations.append(observation)

observations.sort(key=lambda observation: observation['target'])

plt.plot(range(len(observations)), [observation['actual'] for observation in observations])
# plt.plot(range(len(observations)), [observation['target'] * 2 - 1 for observation in observations])
plt.plot(range(len(observations)), [observation['target'] for observation in observations])
plt.show()
