import pickle
import math
import neat

from pytocl.driver import Driver
from pytocl.car import State, Command, DEGREE_PER_RADIANS, MPS_PER_KMH


class MyDriver(Driver):

    timeSinceLastShift = 0

    gearShiftParameters = {
        'shiftDelay': 25,
        'up': 7500,
        'down': 2000
    }

    offtrack = 0
    recovering = False

    roadmap = []

    def __init__(self, net):

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'myneat/config')

        super().__init__()
        if not net is None: self.net = net
        else:
            with open('winner-neat', 'rb') as file:
                pickled = pickle.load(file)
                self.net = neat.nn.FeedForwardNetwork.create(pickled, config)
                
        self.state = 'normal'

        with open('roadmap', 'wb') as file:
            pickle.dump([], file)

        self.speedup = None

    def drive(self, carstate: State) -> Command:

        if self.speedup is None:
            self.speedup = any(opponent !=200 for opponent in carstate.opponents)
            print(self.speedup)

        command = Command()

        print(self.offtrack, carstate.distance_from_center)
        if(self.recovering): print('recovering')


        if abs(carstate.distance_from_center) > 1 or (carstate.angle > 45 and carstate.angle < 315) or self.recovering:

            self.offtrack += 1

            if self.offtrack > 10:
                self.recovering = True
                self.accelerate(carstate, 20, command)
                self.steer(carstate, 0, command)

                if abs(carstate.distance_from_center) < 1 and (carstate.angle < 45 or carstate.angle > 315):
                    self.recovering = False
                    self.offtrack = 0

                if command.steering < 0: command.steering = max(command.steering, -0.07)
                elif command.steering > 0: command.steering = min(command.steering, 0.07)

                return command

        sample = self.state2sample(carstate)

        result = self.net.activate(sample)

        command.accelerator = self.my_accelerate(carstate.speed_x)
        command.gear = self.shiftGears(carstate.gear, carstate.rpm)
        command.steering = result[0] - 0.5

        command.brake = 0

        interval = 10
        position = math.floor(int(carstate.distance_from_start) / interval)

        with open('roadmap', 'rb') as file:
            self.roadmap = pickle.load(file)

        if self.speedup and position + 1 < len(self.roadmap):
            if max(self.roadmap[position:position + int(200 / interval)]) < 0.1 and carstate.speed_x < 180:
                command.accelerator = 1
            else:
                if carstate.speed_x > 80:
                    command.brake = 1
                    command.accelerator = 0


        if len(self.roadmap) == position and int(carstate.distance_from_start) % interval == 0:
            self.roadmap.append(abs(command.steering))
            print('Pos: {}, len(RM): {}'.format(position, len(self.roadmap)))
            with open('roadmap', 'wb') as file:
                pickle.dump(self.roadmap, file)

        print('brake: {}, acc: {}, steering: {}'.format(command.brake, command.accelerator, command.steering))

        return command

    def state2sample(self, carstate: State):

        sample = []
        sample.append(carstate.angle / DEGREE_PER_RADIANS)
        sample.append(carstate.distance_from_center)
        sample.append(carstate.speed_x / MPS_PER_KMH)
        [sample.append(distance) for distance in carstate.distances_from_edge]

        return sample

    def shiftGears(self, previousGear: int, rpm: float) -> int:

        if (previousGear <= 0):
            newGear = 1
        elif (self.timeSinceLastShift < self.gearShiftParameters['shiftDelay']):
            newGear = previousGear
        else:
            if (rpm > self.gearShiftParameters['up']):
                newGear = min(previousGear + 1, 6)
            elif (rpm < self.gearShiftParameters['down']):
                newGear = max(previousGear - 1, 1)
            else:
                newGear = previousGear

        if (previousGear != newGear):
            self.timeSinceLastShift = 0
        else:
            self.timeSinceLastShift += 1

        return newGear

    def my_accelerate(self, speed) -> float:

        # full acceleration below 20 km/h, 0.2 at 80 km/h, linearly less in between

        if speed < 20: return 1
        else: return min(0.2, -0.02 * speed + 1.8)

