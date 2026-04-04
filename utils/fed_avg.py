import copy
import torch


def fed_avg(global_model, client_states):
    """Average client state dicts and load into global_model (in-place).

    This implementation moves tensors to CPU for stable aggregation and supports differing devices/dtypes.
    """
    new_state = copy.deepcopy(global_model.state_dict())
    num_clients = len(client_states)
    if num_clients == 0:
        return global_model

    for key in new_state.keys():
        stacked = torch.stack([client_states[i][key].float().cpu() for i in range(num_clients)], dim=0)
        averaged = torch.mean(stacked, dim=0)
        # cast back to original dtype
        new_state[key] = averaged.type(new_state[key].dtype)

    global_model.load_state_dict(new_state)
    return global_model
