import copy
import torch

def fed_avg(global_model, client_states):
    new_state = copy.deepcopy(global_model.state_dict())
    num_clients = len(client_states)

    for key in new_state.keys():
        new_state[key] = sum(client_states[i][key] for i in range(num_clients)) / num_clients

    global_model.load_state_dict(new_state)
    return global_model
