import pickle as pkl


def check_type(spkr):
    thres = 724
    if int(spkr[1:]) < thres:
        return "valid"
    return "test"

dir = "/mnt/E/arthur.wang/aishell/aishell1/test_pair/logfbank/mix"

frames = [100, 200, 400, 800, 1200, 1600, 3000]


for fr in frames:
    path = f"{dir}/{fr}.pkl"

    with open(path, "rb") as f:
        data = pkl.load(f)
    
    same = {"valid": 0, "test": 0}
    diff = {"valid": 0, "test": 0, "mix": 0}

    for line in data:
        raw_label, gender_label, spkr_1, spot_1, spkr_2, spot_2 = line
        
        t1 = check_type(spkr_1)
        t2 = check_type(spkr_2)

        if spkr_1 == spkr_2:
            same[t1] += 1
        else:
            if t1 == t2:
                diff[t1] += 1
            else:
                diff["mix"] += 1
    print(same)
    print(diff)

        