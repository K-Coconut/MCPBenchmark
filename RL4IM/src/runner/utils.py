import os
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import torch


def load_checkpint(args, runner, agent):
    if args.checkpoint_path != None:

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            runner.logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        runner.logger.console_logger.info("Loading model from {}".format(model_path))
        agent.load_model(model_path)
        agent.global_t = timestep_to_load


def get_inv(m):
    '''

    Parameters
    ----------
    m: 2-dimensions diagnosis matrix torch.sparse_coo_tensor

    Returns
    -------
    inv of m, 2-dimensions torch.sparse_coo_tensor

    '''
    v = m.coalesce().values()
    nbr_node = m.shape[0]
    inv_m = inv(csc_matrix((v, (range(nbr_node), range(nbr_node))), (nbr_node, nbr_node)))
    inv_m = inv_m.tocoo()
    return torch.sparse_coo_tensor((inv_m.row, inv_m.col), inv_m.data)
