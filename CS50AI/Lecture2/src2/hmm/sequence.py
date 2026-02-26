from model import model
import torch

# Observations: 0=umbrella, 1=no umbrella
obs_map = {"umbrella": 0, "no umbrella": 1}
state_names = ["sun", "rain"]

observations = [
    "umbrella",
    "umbrella",
    "no umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "no umbrella",
    "no umbrella",
]

# Convert to tensor of shape [1, sequence_length, 1]
X = torch.tensor([[obs_map[o]] for o in observations], dtype=torch.int32).unsqueeze(0)

# Predict underlying states
predictions = model.predict(X)[0].tolist()
for prediction in predictions:
    print(state_names[prediction])
