'''
standard lstm
'''

import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable

import util
import copy


###################################################################################################
'''
the overall network
'''
class baseline1_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline1_net, self).__init__()

        self.num_class = data_params['n_class']
        self.num_feats = data_params['n_feats']
        self.num_layers = hyperparams['n_layer']
        self.layer_size = hyperparams['layer_s']
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=2)
        
        self.hidden = nn.LSTM(self.num_feats, self.layer_size, num_layers=self.num_layers, batch_first=True)
        self.output = nn.LSTM(self.layer_size, self.num_class, batch_first=True)
        self.rec_layers = [self.hidden, self.output]
           
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.rec_layers)):
            for name, param in self.rec_layers[i].named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
    
    def forward(self, inp, lens, return_extra=False, get_grad=True):
        inp_for_grads = nn.utils.rnn.pad_sequence(copy.deepcopy(inp), batch_first=True)
        out1 = nn.utils.rnn.pad_sequence(inp, batch_first=True)
        if get_grad:
            out1 = Variable(out1).requires_grad_(True)
        for i in range(len(self.rec_layers) - 1):
            out = nn.utils.rnn.pack_padded_sequence(out1, lens, batch_first=True, enforce_sorted=False)
            out = self.rec_layers[i](out)[0]
            out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
            out = self.activation(out)
        
        preds = nn.utils.rnn.pack_padded_sequence(out, lens, batch_first=True, enforce_sorted=False)
        preds = self.rec_layers[-1](preds)[0]
        preds = nn.utils.rnn.pad_packed_sequence(preds, batch_first=True)[0]
        preds = self.softmax(preds)
        
        out = out
        preds = preds
        
        if return_extra:
            return {'input': inp, 'lens': lens, 'preds': preds, 'emb': out, \
                    'inp_for_plots': inp_for_grads}
        
        return preds[np.arange(len(inp)), lens - 1, :]
    
    def get_parameters(self):
        net_params = [sub.parameters() for sub in self.rec_layers]
        return itertools.chain.from_iterable(net_params)
    

###################################################################################################
'''
cross entropy
'''
class baseline1_loss(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline1_loss, self).__init__()
        self.long_thresh = hyperparams['long']
        self.weights = torch.Tensor(hyperparams['weight'])
        
        self.loss = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, outputs, labs):
        lens = outputs['lens']
        preds = torch.squeeze(outputs['preds'])
        
        flat_preds = preds[np.arange(len(lens)), lens - 1, :] 

        return self.loss(flat_preds, labs.type(torch.LongTensor))


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
