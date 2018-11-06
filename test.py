import pickle

from numpy import dot, exp, random

import gym


class NeuralNetwork:
    def __init__(self, number_of_layers, weights):
        self.number_of_layers = number_of_layers
        self.weights = weights
    
    def sigmoid(self, X):
        """ activation function """
        return 1 / (1 + exp(-X))
    
    def feed_forward(self, X):
        """ finds output of neural network provided input """
        l = X
        for i in range(self.number_of_layers - 1):
            w = self.weights[i]
            l = self.sigmoid(dot(w.T, l))
        self.y = l
        return self.y

# loading model from saved file
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

score = data['score']
number_of_layers = data['number_of_layers']
weights = data['weights']

print(score)

# creating neural network with stored weights
nn = NeuralNetwork(number_of_layers, weights)

# running gym environment
done = False                    # flag set by gym when game is over
# creating environment
env = gym.make('CartPole-v0')   
X = env.reset()     

while not done:
    env.render()
    y = 1 if nn.feed_forward(X)[0] > 0.5 else 0
    X, reward, done, info = env.step(y)
