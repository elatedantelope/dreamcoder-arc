import json
import numpy as np

#first, apply the dropping
#then, apply the rigid transformations

# a function for applying the proper rigid transformation to a grid, based on the degrees of rotation (90,180,270) and whether it should be mirrored (True,False)
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


# function to drop the testing grid and create a new task out of each of the training grids
def test_drop():
    # iniatilize an empty dictionary in which to store the new tasks
    drop_dict = {"train": {}, "eval": {}}
    # iterate through each task in the training data
    for id in data["train"]:
        # check to see if there's more than one training grid for this task (if there's only one then we won't do the replacement because then we turn the only training grid into a testing grid and we don't have anything for training)
        if len(data["train"][id]["train"])>1:
            # iterate through each training grid of the task, so there are as many new tasks as there are training grids
            for i in range(len(data["train"][id]["train"])):
                # create a new id for this new task, by concatonating the padded iterator
                idstr = id + f"{i}".zfill(2)
                # initialize an empty dictionary in which to store the training and testing grids for this new task
                newtask = {}
                # copy the testing and training grids onto the new dictionary
                newtask["train"] = data["train"][id]["train"].copy()
                newtask["test"] = data["train"][id]["test"].copy()
                # delete the (old) testing grid
                newtask["test"] = []
                # add the new testing grid, by copying from the current (old) training grid
                newtask["test"].append(data["train"][id]["train"][i])
                # remove the corresponding old training grid from the new dictionary
                newtask["train"].pop(i)
                # we now have a 
                # add the new dictionary to the outer dictionary for storing the new task by its new id
                drop_dict["train"][idstr] = newtask
    return(drop_dict)

dropped_data = test_drop()

# uncomment this if you want to separately write the drop transformations to one file
# write the drop transformations to the json
#with open("transformations.json", "w") as outfile:
    #json.dump(dropped_data, outfile)


outer_dict = {"train": {}, "eval": {}}

def augment(id, rotation, mirror):
    aug_dict = {"train": [], "test": []}
    for item in range(len(dropped_data["train"][id]["train"])):
        input_array = np.array(dropped_data["train"][id]["train"][item]["input"])
        output_array = np.array(dropped_data["train"][id]["train"][item]["output"])
        aug_dict["train"].append({"input": rotate(input_array, rotation, mirror), "output": rotate(output_array, rotation, mirror)})
    for item in range(len(dropped_data["train"][id]["test"])):
        input_array = np.array(dropped_data["train"][id]["test"][item]["input"])
        output_array = np.array(dropped_data["train"][id]["test"][item]["output"])
        aug_dict["test"].append({"input": rotate(input_array, rotation, mirror), "output": rotate(output_array, rotation, mirror)})
    return aug_dict

'''
### uncomment this if you want to separately read the transformations from another file

with open('transformations.json', 'r') as file:
    data = json.load(file)

outer_dict = {"train": {}, "eval": {}}
# a function for generating a new task from a given task, based on the arguments
# arg 1: "train" or "eval", to choose which part of the dataset the given task is from
# arg 2: id e.g. "a68b268e", the id of the given task
# arg 3: degrees e.g. 90, 180, 270, how many degrees to rotate
# arg 4: True/False, whether it should also be rotated or not
def augment(id, rotation, mirror):
    aug_dict = {"train": [], "test": []}
    for item in range(len(data["train"][id]["train"])):
        input_array = np.array(data["train"][id]["train"][item]["input"])
        output_array = np.array(data["train"][id]["train"][item]["output"])
        aug_dict["train"].append({"input": rotate(input_array, rotation, mirror), "output": rotate(output_array, rotation, mirror)})
    for item in range(len(data["train"][id]["test"])):
        input_array = np.array(data["train"][id]["test"][item]["input"])
        output_array = np.array(data["train"][id]["test"][item]["output"])
        aug_dict["test"].append({"input": rotate(input_array, rotation, mirror), "output": rotate(output_array, rotation, mirror)})
    return aug_dict
'''

# takes a task (id) as input, applies rotations, adds them to dictionary with new ids
def rotations():
    for id in dropped_data["train"]:
        # first, apply the rotations without mirroring
        degrees = 90
        for letter in ['A','B','C']:
            outer_dict["train"][f"{id}{letter}"] = augment(id, degrees, False)
            degrees+=90
        #then, apply the rotations with mirroring
        degrees = 0
        for letter in ['D','E','F','G']:
            outer_dict["train"][f"{id}{letter}"] = augment(id, degrees, True)
            degrees+=90
    return(outer_dict)

# apply the rotations to each task in the training set

with open("all_augmentations.json", "w") as outfile:
    json.dump(rotations(), outfile)
