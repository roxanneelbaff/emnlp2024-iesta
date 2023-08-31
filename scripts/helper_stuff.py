import glob
import pickle

PATH_TO_ACTIVATION_STORAGE = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/steering_vectors/eacl_iesta/7b-chat/"

activation_files = glob.glob(f"{PATH_TO_ACTIVATION_STORAGE}*")

for file in activation_files:
    print(file.split("/")[-1])
    if file.split("/")[-1] != 'conservative_ineffective_activations_2.pkl':
        continue
    with open(file,'rb') as f:
        activations = pickle.load(f)
        print(len(activations))
        print(activations[-2])
        for acti in activations[2:]:
            print(acti)
            exit()