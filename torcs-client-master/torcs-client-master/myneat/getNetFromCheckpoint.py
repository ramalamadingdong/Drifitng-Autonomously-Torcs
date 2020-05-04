import pickle
from collections import namedtuple

# from neat.checkpoint import Checkpointer

# population = Checkpointer().restore_checkpoint('neat-checkpoint-222')
#
# Dummy = namedtuple('Dummy', ['fitness'])
#
# winner = Dummy(fitness=0)
#
# for index in population.population.keys():
#     individual = population.population[index]
#     print(individual.fitness)
#     if(individual.fitness and individual.fitness > winner.fitness):
#         winner = individual
#
# with open('winner-neat-full-2', 'wb') as file:
#     pickle.dump(winner, file)

config = Config(
    DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, 'myneat/config'
)

with open('winner-neat', 'rb') as file:
    # unpickler = pickle.Unpickler(file)
    # pickled = unpickler.load()
    pickled = pickle.load(file)
    net = FeedForwardNetwork.create(pickled, config)

with open('winner-neat-net', 'wb') as file:
    pickle.dump(net, file)