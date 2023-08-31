# adapted from emex-emotion-explanation-in-ai/-/blob/main/scripts/training/classification_with_steering_vectors_goemo.py
import os
os.environ["OMP_NUM_THREADS"]="32" # has to be done before any package is imported

import glob
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# VECTOR_TYPE = "steering"
VECTOR_TYPE = "activations"
ACTI_COMPARE_VECS = "all"
# ACTI_COMPARE_VECS = "fair"

if VECTOR_TYPE == "steering":
    print("## NOT SUPPORTED RIGHT NOW ##")
    exit(-1)
elif VECTOR_TYPE == "activations":
    print("## LOADING ACTIVATION VECTORS ##")    
else:
    print("Options for VECTOR_TYPE are -steering- or -activations-")
    exit(-1)

### LOADING ACTIVATIONS

FILES = glob.glob("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0003/emex/steering_vectors/eacl_iesta/7b-chat/*.pkl")

activations = {"conservative_effective" : [], "conservative_ineffective" : [],
                "liberal_effective" : [], "liberal_ineffective" : []}

for file in FILES:
    for key in activations.keys():
        if key in file:
            with open(file, 'rb') as f:
                actis = pickle.load(f)
                activations[key] = activations[key] + actis



# print(len(activations[keys[0][0]]))
# exit()

# taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_classification(y_train,y_test,y_score, n_classes, target_names, layer_indices):
    from itertools import cycle
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import RocCurveDisplay

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  # (n_samples, n_classes)
    fig, ax = plt.subplots(figsize=(6, 6))

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        #label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "purple", "green"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            #name=f"ROC curve for {target_names[class_id]}",
            name=f"{target_names[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--") #, label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel("False Positive Rate", fontsize = 15)
    plt.ylabel("True Positive Rate", fontsize = 15)
    # plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend(loc = "lower right", fontsize = 13)
    fig_name_indices = ""
    for layer_idx in layer_indices:
        fig_name_indices += f"{layer_idx}_"
    fig_name = f"ROC_goemo_{fig_name_indices}steering.pdf" if VECTOR_TYPE == "steering" else f"ROC_{fig_name_indices}actis_{ACTI_COMPARE_VECS}.pdf"
    # fig_name = f"test.pdf"
    plt.savefig(f"images/{fig_name}")
    plt.clf()

# logistic regression iterating over a specified layer
def single_layer_classification(layer_index):
    """Training a logistic regression classifier with single layers as input. 

    :param int layer_index: Which layer should be used.
    """
    
    if (VECTOR_TYPE == "activations") and (ACTI_COMPARE_VECS == "all"):
        Y_train = []
        X_train = []
        for entry in go_emo_train:
            Y_train.append(labels.index(entry[1]['labels'][0]))
            X_train.append(entry[2][layer_index])

        Y_test = []
        X_test = []
        for entry in go_emo_test:
            Y_test.append(labels.index(entry[1]['labels'][0]))
            X_test.append(entry[2][layer_index])

    else:
        Y_25, Y_17, Y_14, Y_2, Y_26, Y_11 = [],[],[],[],[],[]
        X_25, X_17, X_14, X_2, X_26, X_11 = [],[],[],[],[],[]
        
        entry_list = go_emo_train_actis_fair

        for entry in entry_list:
            class_label = entry[1]['labels'][0]

            if class_label == 25:
                Y_25.append(labels.index(entry[1]['labels'][0]))
                X_25.append(entry[2][layer_index-18])
            elif class_label == 17:
                Y_17.append(labels.index(entry[1]['labels'][0]))
                X_17.append(entry[2][layer_index-18])
            elif class_label == 14:
                Y_14.append(labels.index(entry[1]['labels'][0]))
                X_14.append(entry[2][layer_index-18])
            elif class_label == 2:
                Y_2.append(labels.index(entry[1]['labels'][0]))
                X_2.append(entry[2][layer_index-18])
            elif class_label == 26:
                Y_26.append(labels.index(entry[1]['labels'][0]))
                X_26.append(entry[2][layer_index-18])
            elif class_label == 11:
                Y_11.append(labels.index(entry[1]['labels'][0]))
                X_11.append(entry[2][layer_index-18])
            else:
                print(f"Didn't find {class_label}")

        X_train, X_test = [],[]
        Y_train, Y_test = [], []
        split_ratio = .5
        for tup in [(X_25,Y_25), (X_17,Y_17), (X_14,Y_14), (X_2,Y_2), (X_26,Y_26), (X_11,Y_11)]:
            end_train_idx = int(split_ratio * len(tup[0]))+1
            X_train.extend(tup[0][0:end_train_idx])
            Y_train.extend(tup[1][0:end_train_idx])
            X_test.extend(tup[0][end_train_idx:-1])
            Y_test.extend(tup[1][end_train_idx:-1])
                

    clf = LogisticRegression(multi_class='multinomial', max_iter = 20000, class_weight='balanced').fit(X_train, Y_train)
    print(f"Layer {layer_index} classification score: {clf.score(X_test,Y_test)}")
    plot_classification(Y_train,Y_test, clf.predict_proba(X_test), 6, ["sadness", "joy", "fear", "anger", "surprise", "disgust"], [layer_index])


# logistic regression with concatenated layers, sliding window
def multi_layer_classification(num_layers = 3, specific_layers = None, keys = ['conservative_effective', 'conservative_ineffective']):
    """Training a logistic regression classifier with multiple layers as input. 
    Currently it only works for the activation-based vectors.

    :param int num_layers: How many layers per classifier, defaults to 3
    :param array specific_layers: Which layers should be used , defaults to None
    :param dict keys: Which classes should be compared, defaults to ['conservative_effective', 'conservative_ineffective']
    """
    
    layer_indices_list = []
    if specific_layers is not None:
        layer_indices_list = [specific_layers] 
    else:
        for i in range(0,len(activations[keys[0]][0][2])):
            layer_indices_list.append(np.arange(i,i+num_layers))

    for layer_indices in layer_indices_list:
        
        class_one_actis = activations[keys[0]]
        np.random.shuffle(class_one_actis)
        class_two_actis = activations[keys[1]]
        np.random.shuffle(class_two_actis)

        # print(class_one_actis)
        # print(class_two_actis)
        # print(len(activations[keys[0][0]]))
        # balancing dataset
        sample_size = len(class_one_actis) if len(class_one_actis) < len(class_two_actis) else len(class_two_actis)
        class_one_samples = np.random.choice(class_one_actis, size = sample_size, replace=False)
        class_two_samples = np.random.choice(class_two_actis, size = sample_size, replace=False)
        # splitting train/test set
        class_one_train = class_one_samples[:int(len(class_one_samples) * 0.75)]
        class_two_train = class_two_samples[:int(len(class_two_samples) * 0.75)]
        class_one_test = class_one_samples[int(len(class_one_samples) * 0.75):]
        class_two_test = class_two_samples[int(len(class_two_samples) * 0.75):]

        Y_train = []
        X_train = []
        for entry in class_one_train:
            sample = entry[2]
            Y_train.append(0)
            entries = []
            for layer_index in layer_indices:
                entries.append(sample[layer_index])
                # entries.append(entry[0][layer_index])
            X_train.append(np.concatenate(entries))

        for entry in class_two_train:
            sample = entry[2]
            Y_train.append(1)
            entries = []
            for layer_index in layer_indices:
                entries.append(sample[layer_index])
                # entries.append(entry[0][layer_index])
            X_train.append(np.concatenate(entries))

        Y_test = []
        X_test = []
        for entry in class_one_test:
            sample = entry[2]
            Y_test.append(1)
            entries = []
            for layer_index in layer_indices:
                entries.append(sample[layer_index])
                # entries.append(entry[0][layer_index])
            X_test.append(np.concatenate(entries))

        for entry in class_two_test:
            sample = entry[2]
            Y_test.append(1)
            entries = []
            for layer_index in layer_indices:
                entries.append(sample[layer_index])
                # entries.append(entry[0][layer_index])
            X_test.append(np.concatenate(entries))

        clf = LogisticRegression(multi_class='multinomial', max_iter = 10000, class_weight='balanced').fit(X_train, Y_train)
        print(f"Layer {layer_indices[0]} classification score: {clf.score(X_test,Y_test)}")
        plot_classification(Y_train,Y_test, clf.predict_proba(X_test), 2, ["conservative effective", "conservative ineffective"], layer_indices)

multi_layer_classification(num_layers = 1)
# single_layer_classification(18)
# single_layer_classification(19)
# single_layer_classification(20)
