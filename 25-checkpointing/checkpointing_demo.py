import json

state = {
    "step": 1,
    "message": "Processing started"
}

with open("checkpoint.json", "w") as f:
    json.dump(state, f)

with open("checkpoint.json") as f:
    restored_state = json.load(f)

print(restored_state)

