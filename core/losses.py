from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.MSE_Fn = nn.MSELoss() 


    def forward(self, out, label):

        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])
        l_cl = out['token_con']

        loss = self.alpha * l_cl +  l_sp

        return {'loss': loss, 'l_sp': l_sp, 'l_cl': l_cl}

