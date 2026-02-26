from model import model
from collections import Counter

# Variable order: [rain, maintenance, train, appointment]
# Indices:
#   rain:        0=none, 1=light, 2=heavy
#   maintenance: 0=yes,  1=no
#   train:       0=on time, 1=delayed
#   appointment: 0=attend,  1=miss

rain_labels = ["none", "light", "heavy"]
maintenance_labels = ["yes", "no"]
train_labels = ["on time", "delayed"]
appointment_labels = ["attend", "miss"]


def generate_sample():
    # Sample from the model (returns tensor of shape [1, 4])
    s = model.sample(1)[0].tolist()
    return {
        "rain": rain_labels[s[0]],
        "maintenance": maintenance_labels[s[1]],
        "train": train_labels[s[2]],
        "appointment": appointment_labels[s[3]],
    }


# Rejection sampling
# Compute distribution of Appointment given that train is delayed
N = 10000
data = []
for i in range(N):
    sample = generate_sample()
    if sample["train"] == "delayed":
        data.append(sample["appointment"])

print(Counter(data))
