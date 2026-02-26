from model import model
import torch

# Variable order: [rain, maintenance, train, appointment]
# Indices:
#   rain:        0=none, 1=light, 2=heavy
#   maintenance: 0=yes,  1=no
#   train:       0=on time, 1=delayed
#   appointment: 0=attend,  1=miss

# Observation: rain=none, maintenance=no, train=on time, appointment=attend
X = torch.tensor([[0, 1, 0, 0]], dtype=torch.int32)

probability = model.probability(X)

print(probability)
