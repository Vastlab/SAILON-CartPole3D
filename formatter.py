import numpy as np


def format_istate_data(feature_vector):
    # Format data for use with evm
    # print(feature_vector)
    cur_state = []
    for i in feature_vector.keys():
        if i == 'time_stamp' or i == 'image' or i == 'hint':
            continue
        if i != 'blocks' and i != 'walls':
            # print(feature_vector[i])
            for j in feature_vector[i]:
                # print(j)
                cur_state.append(feature_vector[i][j])
        elif i == 'blocks':
            for block in feature_vector[i]:
                for key in block.keys():
                    if key != 'id':
                        cur_state.append(block[key])
        elif i == 'walls':
            # Add in data for the walls
            for j in feature_vector[i]:
                for k in j:
                    cur_state.append(k)

    return np.asarray(cur_state)


def format_data_without_blocks(feature_vector):
    # Format data for use with evm
    state = []
    for i in feature_vector.keys():
        #            if i != 'blocks' and i != 'time_stamp' and i != 'image' and i != 'ticks':
        if i == 'cart' or i == 'pole':
            #                print(i,feature_vector[i])
            for j in feature_vector[i]:
                state.append(feature_vector[i][j])
            # print(state)
        elif i == 'blocks':
            for block in feature_vector[i]:
                for key in block.keys():
                    if key != 'id':
                        state.append(block[key])

    return np.asarray(state)


def format_data(feature_vector):
    # Format data for use with evm
    state = []
    for i in feature_vector.keys():
        if i == 'cart' or i == 'pole':
            for j in feature_vector[i]:
                state.append(feature_vector[i][j])
            # print(state)
    return np.asarray(state)
