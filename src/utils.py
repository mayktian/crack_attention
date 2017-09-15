def _lengths_to_masks(lengths, max_length):
    
    tiled_ranges = autograd.Variable(torch.arange(0,float(max_length)).unsqueeze(0).expand([len(lengths),max_length]))
    
    lengths = lengths.float().unsqueeze(1).expand_as(tiled_ranges)
    
    mask = tiled_ranges.lt(lengths).float()

    return mask

def weight_variable(shape):
    
    initial = np.random.uniform(-0.01, 0.01,shape)
    
    initial = torch.from_numpy(initial)
    
    return initial.float()