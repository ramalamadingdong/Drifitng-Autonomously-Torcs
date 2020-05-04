import os

from pandas import DataFrame, Series
from torch.utils.data import Dataset
import sqlite3

observation = {}
keys = [
    'accel', 'brake', 'gear', 'gear2', 'steer', 'clutch', 'curTime', 'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 'fuel', 'lastLapTime', 'racePos', 'opponents0', 'opponents1', 'opponents2', 'opponents3', 'opponents4', 'opponents5', 'opponents6', 'opponents7', 'opponents8', 'opponents9', 'opponents10', 'opponents11', 'opponents12', 'opponents13', 'opponents14', 'opponents15', 'opponents16', 'opponents17', 'opponents18', 'opponents19', 'opponents20', 'opponents21', 'opponents22', 'opponents23', 'opponents24', 'opponents25', 'opponents26', 'opponents27', 'opponents28', 'opponents29', 'opponents30', 'opponents31', 'opponents32', 'opponents33', 'opponents34', 'opponents35', 'rpm', 'speedX', 'speedY', 'speedZ', 'track0', 'track1', 'track2', 'track3', 'track4', 'track5', 'track6', 'track7', 'track8', 'track9', 'track10', 'track11', 'track12', 'track13', 'track14', 'track15', 'track16', 'track17', 'track18', 'trackPos', 'wheelSpinVel0', 'wheelSpinVel1', 'wheelSpinVel2', 'wheelSpinVel3', 'z', 'focus0', 'focus1', 'focus2', 'focus3', 'focus4'
]

db = sqlite3.connect('training-data/trainingData.db')

sql = 'DROP TABLE IF EXISTS observations'
db.execute(sql)
db.commit()

sql = 'CREATE TABLE observations (track'
for key in keys:
    sql += (', ' + key)
sql += ');'

db.execute(sql)
db.commit()

for fileName in os.listdir('training-data/extended/'):

    with open('training-data/extended/' + fileName) as file:
        lines = file.readlines()

    print('Reading ' + fileName)

    for line in lines:
        line = line.strip()[1:-1]

        sqlInsert = 'INSERT INTO observations (track, '
        sqlValues = "VALUES ('" + fileName + "', "

        for tuple in line.split(')('):
            valueList = tuple.split()
            key = valueList.pop(0)

            if len(valueList) == 1:
                sqlInsert += key + ', '
                sqlValues += valueList[0] + ', '
            else:
                for i, value in enumerate(valueList):
                    sqlInsert += key + str(i) + ', '
                    sqlValues += value + ', '

        sql = sqlInsert[:-2] + ') ' + sqlValues[:-2] + ')'
        db.execute(sql)

    db.commit()