import numpy as np
from copy import deepcopy
from .nn import NN
from .aibird import AIBird

# Takes a child and 2 parents and selects elements from parents to give to child
# Used to merge 2 best performing parents to form new child
def crossover(parent_a: NN, parent_b: NN) -> NN:
    child = NN()
    for attr in ('w1', 'b1', 'w2', 'b2'):
        pa = getattr(parent_a, attr)
        pb = getattr(parent_b, attr)
        mask = np.random.rand(*pa.shape) > 0.5
        setattr(child, attr, np.where(mask, pa, pb))
    return child

# Used to explore new behaviors by adding randomness to weights
# Each weight has 10% chance of getting noise
def mutate(net: NN, rate=0.1, scale=0.5)->NN:
    for attr in ('w1', 'b1', 'w2', 'b2'):
        mat = getattr(net, attr)
        mutation_mask = np.random.rand(*mat.shape) < rate
        noise = np.random.randn(*mat.shape) * scale
        mat += mutation_mask * noise
        setattr(net, attr, mat)
    return net

# Builds next generation of birds
def evolve_population(birds: list[AIBird], config, retain_frac=0.2, random_frac=0.05) -> list[AIBird]:
    # Sort birds and keep the elite birds in parents array (top performing)
    birds.sort(key=lambda b: b.fitness, reverse=True)
    retain_len = int(len(birds) * retain_frac)
    parents = birds[:retain_len]
    
    # Keep low performers at times for genetic diversity
    for bird in birds[retain_len:]:
        if np.random.rand() < random_frac:
            parents.append(bird)
    
    # Crossover + mutate to refill population
    children = []
    desired = len(birds) - len(parents)
    while len(children) < desired:
        pa, pb = np.random.choice(parents, 2, replace=False)
        child_net = crossover(pa.model, pb.model)
        mutate(child_net)
        child = AIBird.from_model(child_net, config)
        children.append(child)
    return parents + children