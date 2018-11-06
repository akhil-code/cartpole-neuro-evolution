import pickle

import gym
from genetic import Population


class Game:
    def __init__(self):
        # layers used for neural network
        self.layers = (4, 64, 64, 64, 1)
        # initializing population for genetic algorithm
        self.population = Population(layers=self.layers)
        # initializing openai gym environment for cartpole game
        self.env = gym.make('CartPole-v0')
    
    def loop(self):
        """ loops infinitely for training """
        high_score = 0                          # used to store high score in training so far
        # learn infinitely
        while True:
            # iterating individuals in population
            for individual in self.population.individuals:
                X = self.env.reset()            # Feature vector
                done = False                    # flag set by gym when game is over

                while not done:
                    # self.env.render()
                    y = 1 if individual.nn.feed_forward(X)[0] > 0.5 else 0
                    X, reward, done, info = self.env.step(y)
                    individual.score += reward
                
                # save model when new high score is achieved
                if individual.score > high_score:
                    data = {
                        'score' : individual.score,
                        'number_of_layers' : individual.nn.number_of_layers,
                        'weights' : individual.nn.weights,
                    }
                    # write to file
                    with open('model.pkl', 'wb') as f:
                        pickle.dump(data, f)
                    high_score = individual.score
            
            self.population.evolve()

if __name__ == '__main__':
    game = Game()
    game.loop()
