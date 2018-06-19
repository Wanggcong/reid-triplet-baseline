import torch as th
import numpy as np
def _shuttle(inputs, labels):
    ind = np.random.choice(inputs.size(0),inputs.size(0),replace=False).tolist()
    ind = th.LongTensor(ind)
    inputs_rand = th.index_select(inputs, 0, ind)
    labels_rand = th.index_select(labels, 0, ind)
    
    # print('inputs rand:', inputs.type())
    # print('labels rand:', labels.type())

    # print('inputs rand:', inputs_rand.type())
    # print('labels rand:', labels_rand.type())
    # input('hi, wgc!')
    inputs_new = th.cat((inputs[:,:,0:128,:],inputs_rand[:,:,128:256,:]),2)
    labels = labels*100+labels_rand

    return inputs_new, labels