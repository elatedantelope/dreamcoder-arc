import json
import numpy as np

def rotate(array, deg, mirror):
    if mirror:
        newarr = np.flip(array, 0)
        if deg == 0:
            rotated = newarr.tolist()
        elif deg == 90:
            rotated = np.rot90(newarr, k=1).tolist()
        elif deg == 180:
            rotated = np.rot90(np.rot90(newarr, k=1)).tolist()
        elif deg == 270:
            rotated = np.rot90(np.rot90(np.rot90(newarr, k=1))).tolist()
        else:
            rotated = "invalid rotation"
    else:
        if deg == 90:
            rotated = np.rot90(array, k=1).tolist()
        elif deg == 180:
            rotated = np.rot90(np.rot90(array, k=1)).tolist()
        elif deg == 270:
            rotated = np.rot90(np.rot90(np.rot90(array, k=1))).tolist()
        else:
            rotated = "invalid rotation"
    return rotated


# Open and read the JSON file
with open('arc1.json', 'r') as file:
    data = json.load(file)

newdict = {"train": {}, "eval": {}}

# function to drop the testing sample and create n new tasks where each new task is the old task with the training sample removed and one of the training samples is elevated to be the testing sample
def test_drop(id):
    # check to see if there's more than one training sample (if there's only one then we won't do the replacement because then we turn the only training sample into a testing sample and we don't have anything for training)
    if len(data["train"][id]["train"])>1:
        # iterate through each training grid, so there are as many new tasks as there are training grids
        for i in range(len(data["train"][id]["train"])):
            idstr = id + f"{i}".zfill(2)
            tmpdict = {}
            # set the testing grid to be the current training grid
            tmpdict["train"] = data["train"][id]["train"].copy()
            tmpdict["test"] = data["train"][id]["test"].copy
            tmpdict["test"] = []
            tmpdict["test"].append(data["train"][id]["train"][i])
            tmpdict["train"].pop(i)
            newdict["train"][idstr] = tmpdict
    return(newdict)

'''
#print("New tasks:\n")

id = "a68b268e"

expansion = test_drop(id)
#print(len(expansion["train"]["a68b268e00"]["train"]))

newid = "8d510a79"

expansion = test_drop(newid)

print(expansion)

#print(expansion)
'''

def new_dropped_dataset():
    for id in data["train"]:
        expansion = test_drop(id)
    return(expansion)

with open("transformations.json", "w") as outfile:
    json.dump(new_dropped_dataset(), outfile)



#outer_dict = {id: data["train"][id]}
#aug_dict = {"train": [], "test": []}

outer_dict = {"train": {}, "eval": {}}

# a function for generating a new task from a given task, based on the arguments
# arg 1: "train" or "eval", to choose which part of the dataset the given task is from
# arg 2: id e.g. "a68b268e", the id of the given task
# arg 3: degrees e.g. 90, 180, 270, how many degrees to rotate
# arg 4: True/False, whether it should also be rotated or not
def augment(train_or_eval, id, rotation, mirror):
    aug_dict = {"train": [], "test": []}
    for item in range(len(data[train_or_eval][id]["train"])):
        input_array = np.array(data[train_or_eval][id]["train"][item]["input"])
        output_array = np.array(data[train_or_eval][id]["train"][item]["output"])
        aug_dict["train"].append({"input": rotate(input_array, rotation, mirror), "output": rotate(output_array, rotation, mirror)})
    for item in range(len(data[train_or_eval][id]["test"])):
        input_array = np.array(data[train_or_eval][id]["test"][item]["input"])
        output_array = np.array(data[train_or_eval][id]["test"][item]["output"])
        aug_dict["test"].append({"input": rotate(input_array, rotation, mirror), "output": rotate(output_array, rotation, mirror)})
    return aug_dict

# go through the whole dataset and do the seven transformations of each task
# def dataset():


# takes a task (id) as input, applies rotations, adds them to dictionary with new ids
def rotations(train_or_eval, id):
    # first, apply the rotations without mirroring
    degrees = 90
    for letter in ['A','B','C']:
        outer_dict[train_or_eval][f"{id}{letter}"] = augment(train_or_eval, id, degrees, False)
        degrees+=90
    #then, apply the rotations with mirroring
    degrees = 0
    for letter in ['D','E','F','G']:
        outer_dict[train_or_eval][f"{id}{letter}"] = augment(train_or_eval, id, degrees, True)
        degrees+=90

# apply the rotations to each task in the training set

# function for creating the new dataset
# just iterate through each of the tasks in the training set
def create_new_dataset():
    for id in data["train"]:
        rotations("train", id)
# don't iterate through the tasks in the eval set, leave it empty
    #for id in data["eval"]:
        #rotations("eval", id)


'''
create_new_dataset()

with open("transformations.json", "w") as outfile:
    json.dump(outer_dict, outfile)

'''