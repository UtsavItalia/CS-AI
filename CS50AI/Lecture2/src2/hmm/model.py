from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
import torch

# States: 0=sun, 1=rain
# Observations: 0=umbrella, 1=no umbrella

# Observation model for each state
sun  = Categorical([[0.2, 0.8]])   # umbrella, no umbrella
rain = Categorical([[0.9, 0.1]])   # umbrella, no umbrella

# Transition matrix
transitions = torch.tensor([
    [0.8, 0.2],  # sun  -> sun, rain
    [0.3, 0.7],  # rain -> sun, rain
])

# Starting probabilities
starts = torch.tensor([0.5, 0.5])

# Create the model
model = DenseHMM(
    distributions=[sun, rain],
    edges=transitions,
    starts=starts,
)
