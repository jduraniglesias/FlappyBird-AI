import matplotlib.pyplot as plt

def plot_fitness(log):
    plt.figure(figsize=(10, 5))
    plt.plot(log['generation'], log['best'], label='Best Fitness')
    plt.plot(log['generation'], log['avg'], label='Average Fitness')
    plt.plot(log['generation'], log['min'], label='Min Fitness', alpha=0.5)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('AI Bird Fitness Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
