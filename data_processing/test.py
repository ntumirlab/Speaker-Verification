import pickle as pkl

path = "/mnt/E/arthur.wang/aishell/aishell1/opsm/train/S0230/BAC009S0230W0488.pkl"

with open(path, 'rb') as f:
    data = pkl.load(f)

print(len(data))