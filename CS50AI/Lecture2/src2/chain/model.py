from pomegranate.markov_chain import MarkovChain
from pomegranate.distributions import Categorical, ConditionalCategorical

# States: 0=sun, 1=rain
labels = ["sun", "rain"]

# Starting probabilities
start = Categorical([[0.5, 0.5]])

# Transition model
transitions = ConditionalCategorical(
    [
        [
            [0.8, 0.2],  # sun -> sun, sun -> rain
            [0.3, 0.7],
        ]  # rain -> sun, rain -> rain
    ]
)

# Create Markov chain
model = MarkovChain([start, transitions])

# Sample 50 states from chain
samples = model.sample(50).tolist()
print([labels[sample[0][0]] for sample in samples])
