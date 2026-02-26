from model import model
import torch

# Variable order: [rain, maintenance, train, appointment]
# Indices:
#   rain:        0=none, 1=light, 2=heavy
#   maintenance: 0=yes,  1=no
#   train:       0=on time, 1=delayed
#   appointment: 0=attend,  1=miss

# Evidence: train=delayed (index 1)
# -1 means unknown
X = torch.tensor([[-1, -1, 1, -1]], dtype=torch.int32)

predictions = model.predict_proba(X)

# Node names and their state labels
nodes = {
    "rain": ["none", "light", "heavy"],
    "maintenance": ["yes", "no"],
    "train": ["on time", "delayed"],
    "appointment": ["attend", "miss"],
}

for node_name, labels in nodes.items():
    probs = predictions[0][list(nodes.keys()).index(node_name)]
    print(f"{node_name}")
    for label, prob in zip(labels, probs):
        print(f"    {label}: {float(prob):.4f}")
