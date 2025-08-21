import numpy as np

# Applies non-linearity to allow ai to learn better
def relu(x):
    return np.maximum(0,x)

# Returns probabilty of if bird should flap or not (if > .5 = flap)
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class NN:
    def __init__(self, input_size=5, hidden_size=6):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros((1,1))
        self.fitness = 0

    def save(self, path: str):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(path: str) -> "NN":
        data = np.load(path, allow_pickle=False)
        net = NN()
        net.w1 = data["w1"]; net.b1 = data["b1"]
        net.w2 = data["w2"]; net.b2 = data["b2"]
        return net
    
    # AI decision maker
    # Step 1: Takes input x (state vector)
    # Step 2: np.dot(x, self.w1) + self.b1 is input -> hidden layer
    # Step 3: Take first layer and apply non-linearity with relu
    # Step 4: np.dot(h, self.w2) + self.b2 is hidden -> output layer
    # Step 5: Take this layer and use sigmoid to get probability to flap
    def forward(self, x):
        x = np.array(x).reshape(1, -1)
        h = relu(np.dot(x, self.w1) + self.b1)
        out = sigmoid(np.dot(h, self.w2) + self.b2)
        return out[0][0]
    
    def debugFitness(self):
        print(self.fitness)