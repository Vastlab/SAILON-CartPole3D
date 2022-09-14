'''
Create data for use in evm with cartpole lookahead method.
Data for evm is [cart, pole, blocks]. Created by the expected-actual
post-step.

Saved data is form: standard dev, mean, smallest 10, largest 10.
'''

import importlib.util
import numpy as np
import torch
import pdb

NUM_ITERS = 5000
NUM_STEPS = 40
TAILSIZE = 10

FILENAME = (f"cartpole_{NUM_ITERS}iters_{NUM_STEPS}steps_no_change_np")



def print_format_data_order(feature_vector):
    # Format data for use with evm
    out=[]
    for i in feature_vector.keys():
        out.append(i)
        if i != 'blocks':
            #print(feature_vector[i])
            for j in feature_vector[i]:
                out.append(j)
    block_data = []
    for block in feature_vector['blocks'][:1]:
        cur_block = []
        for key in block.keys():
            if key != 'id':
                cur_block.append(("B",key))
        block_data.append(cur_block)
        
    out.append(block_data)
    print("Formated data order is",out)
    return 




def format_data(feature_vector, predicted_state):
    # Format data for use with evm
    cur_state = []
    for i in feature_vector.keys():
        if i != 'blocks':
            #print(feature_vector[i])
            for j in feature_vector[i]:
                cur_state.append(predicted_state[i][j] - feature_vector[i][j])
    cnt = 0
    block_data = []
    for block in feature_vector['blocks'][:1]:
        cur_block = []
        for key in block.keys():
            if key != 'id':
                cur_block.append(abs(predicted_state['blocks'][cnt][key]) - abs(block[key]))
        cnt += 1
        block_data.append(cur_block)
    
    block_diff = [0 for i in range(len(block_data[0]))]
    for data in block_data:
        for i in range(len(data)):
            block_diff[i] += data[i]
    for i in block_diff:
        i = i / len(block_data)
        cur_state.append(i)
    return cur_state


def format_data_reset(feature_vector):
    # Format data for use with evm
    #print(feature_vector)
    cur_state = []
    for i in feature_vector.keys():
        if i != 'blocks' and i != 'walls':
            #print(feature_vector[i])
            for j in feature_vector[i]:
                #print(j)
                cur_state.append(feature_vector[i][j])
        elif i == 'blocks':
            for block in feature_vector[i][:1]:
                for key in block.keys():
                    if key != 'id':
                        cur_state.append(block[key])
        elif i == 'walls':
            # Add in data for the walls
            for j in feature_vector[i]:
                for k in j:

                    cur_state.append(k)

    return np.asarray(cur_state)


def analyze_data(data, k):
    formatted_data = np.array([np.std(data, axis=0)])
    #print(formatted_data.shape)
    #print(np.mean(data, axis=0).shape)
    formatted_data = np.append(formatted_data, [np.mean(data, axis=0)], axis=0)

    #print(formatted_data.shape)

    smallVals = data[0:k]
    largeVals = data[0:k]

    for i in data:
        #print(i)
        for j in range(len(i)):
            # Check if val at i, j is less than current val there.
            for x in range(k):
                if i[j] < smallVals[x][j]:

                    temp = smallVals[x][j]

                    smallVals[x][j] = i[j]
                    # Shift vals down
                    for y in range(x + 1, k):
                        temp2 = smallVals[y][j]
                        smallVals[y][j] = temp
                        temp = temp2
                    break

    # Obtain largest vals
    for i in data:
        #print(i)
        for j in range(len(i)):
            # Check if val at i, j is less than current val there.
            for x in range(k):
                if i[j] > largeVals[x][j]:

                    temp = largeVals[x][j]

                    largeVals[x][j] = i[j]
                    # Shift vals down
                    for y in range(x + 1, k):
                        temp2 = largeVals[y][j]
                        largeVals[y][j] = temp
                        temp = temp2
                    break

def format_data_torch(data, k): # Format the data using torch and tensors instead of numpy.
    data = torch.tensor(data)
    formatted_data = torch.std(data, axis=0)
    formatted_data = torch.stack((formatted_data, torch.mean(data, axis=0)), dim=0)
    # Get smallest TAILSIZE
    smallest, indices = torch.topk(data, k, dim=0, largest=False)
    #print(smallest)
    largest, indices = torch.topk(data, k, dim=0)
    formatted_data = torch.cat((formatted_data, smallest), dim = 0)
    formatted_data = torch.cat((formatted_data, largest), dim = 0)

    print(formatted_data.shape)
    return formatted_data


# Create envs
env_location = importlib.util.spec_from_file_location('CartPoleBulletEnv', \
    'cartpolepp/cartpoleplusplus.py')
env_class = importlib.util.module_from_spec(env_location)
env_location.loader.exec_module(env_class)
env1_location = importlib.util.spec_from_file_location('CartPoleBulletEnv', \
    'cartpolepp/UCart.py')
env1_class = importlib.util.module_from_spec(env1_location)
env1_location.loader.exec_module(env1_class)

myconfig = dict()
myconfig['start_zeroed_out'] = False


# Package params here
params = dict()
params['seed'] = 11
params['config'] = myconfig
params['path'] = "cartpolepp"
params['use_img'] = False
params['use_gui'] = False


env = env_class.CartPoleBulletEnv(params)
env1 = env1_class.CartPoleBulletEnv(params)


feature_vector = env.reset()
env1.resetbase()
env1.reset(feature_vector)
action, next_action, predicted_state = env1.get_best_onestep_action(feature_vector)
#print(action)
feature_vector, _, done, _ = env.step(action)
#print(predicted_state)
prediction = format_data(feature_vector, predicted_state)
data = np.array([prediction])

totalSteps = 0
print_format_data_order(feature_vector)
failcnt=0

for k in range(NUM_ITERS):
    if k != 0:
        feature_vector = env.reset()
        env1.reset(feature_vector)
        #instance_data = []
    # Step envs


        
    for i in range(NUM_STEPS):
        pdb.set_trace()
    #if i % 2 == 0:
        env1.reset(feature_vector)        
        action, next_action, predicted_state = env1.get_best_action(feature_vector)
        feature_vector, _, done, _ = env.step(action)
        print(predicted_state)
        prediction = format_data(feature_vector, predicted_state)
        print("Action", action,"pred",prediction)
       
        '''else:
            print(next_action)
            feature_vector, _, done, _ = env.step(next_action)

            prediction = format_data(feature_vector, predicted_state)'''

        # Save data
        #instance_data.append(prediction)
        data = np.append(data, [prediction], axis=0)
        totalSteps += 1
        if done:
            failcnt += 1
            print(f"Short run {i} steps. fail is {failcnt}, on iteration k={k} ")
            break

    #data.append(list(instance_data))
    #print("Data being saved: ", data)
print(f"Success on {NUM_ITERS-failcnt} fail on {failcnt}  {NUM_ITERS}  for  {100*(1- failcnt/NUM_ITERS)}  percent robustness.")    

#data = np.array(data)
#print(data.shape)
#print("Average steps: ", totalSteps / NUM_ITERS)

iter_data = format_data_torch(data, TAILSIZE)
#iter_data.numpy()
#print("iter_data: ", iter_data.shape)

np.save(FILENAME, iter_data.numpy())
np.save("fulldata-dynamic.npy", data)


# Collect 3000 starts data
FILENAME = ("cartpole_3000_resets_np")
data = np.array([format_data_reset(env.reset())])
print(data.shape)
totalSteps = 0
cnt = 0
while cnt < 3000:
    feature_vector = env.reset()
    #if env.nb_blocks == BLOCK_NUM:
    instance_data = format_data_reset(feature_vector)
    data = np.append(data, [instance_data], axis=0)
    cnt += 1
        #print("Data being saved: ", data)

data = np.array(data)
print(data.shape)

reset_data = format_data_torch(data, TAILSIZE)

#print(reset_data.shape)

np.save(FILENAME, reset_data.numpy())
np.save("fulldata-static.npy", data)

# Save npy file
#np.save(FILENAME, data)
