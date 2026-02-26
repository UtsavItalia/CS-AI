from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
import torch

# Rain: 0=none, 1=light, 2=heavy
rain = Categorical([[0.7, 0.2, 0.1]])

# Maintenance: 0=yes, 1=no
maintenance = ConditionalCategorical(
    [
        [
            [0.4, 0.6],  # rain=none
            [0.2, 0.8],  # rain=light
            [0.1, 0.9],
        ]  # rain=heavy
    ]
)

# Train: 0=on time, 1=delayed
train = ConditionalCategorical(
    [
        [
            [
                [0.8, 0.2],  # rain=none, maintenance=yes
                [0.9, 0.1],
            ],  # rain=none, maintenance=no
            [
                [0.6, 0.4],  # rain=light, maintenance=yes
                [0.7, 0.3],
            ],  # rain=light, maintenance=no
            [
                [0.4, 0.6],  # rain=heavy, maintenance=yes
                [0.5, 0.5],
            ],
        ]  # rain=heavy, maintenance=no
    ]
)

# Appointment: 0=attend, 1=miss
appointment = ConditionalCategorical(
    [
        [
            [0.9, 0.1],  # train=on time
            [0.6, 0.4],
        ]  # train=delayed
    ]
)

model = BayesianNetwork()
model.add_distributions([rain, maintenance, train, appointment])
model.add_edge(rain, maintenance)
model.add_edge(rain, train)
model.add_edge(maintenance, train)
model.add_edge(train, appointment)

# Query: [rain, maintenance, train, appointment], -1 = unknown
X = torch.tensor([[2, -1, -1, 1]], dtype=torch.int32)  # rain=heavy, appointment=miss
print(model.probability(X))
