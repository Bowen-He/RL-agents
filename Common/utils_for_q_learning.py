import torch


class Reshape(torch.nn.Module):
    """
	Description:
		Module that returns a view of the input which has a different size    Parameters:
		- args : Int...
			The desired size
	"""
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def __repr__(self):
        s = self.__class__.__name__
        s += '{}'.format(self.shape)
        return s

    def forward(self, x):
        return x.view(*self.shape)


def sync_networks(target, online, alpha, copy=False):
    if copy == True:
        for online_param, target_param in zip(online.parameters(),
                                              target.parameters()):
            target_param.data.copy_(online_param.data)
    elif copy == False:
        for online_param, target_param in zip(online.parameters(),
                                              target.parameters()):
            target_param.data.copy_(alpha * online_param.data +
                                    (1 - alpha) * target_param.data)

def set_random_seed(meta_params):
    seed_number = meta_params['seed']
    import numpy
    numpy.random.seed(seed_number)
    import random
    random.seed(seed_number)
    import torch
    torch.manual_seed(seed_number)
    meta_params['env'].seed(seed_number)
    meta_params['env'].action_space.np_random.seed(seed_number)
