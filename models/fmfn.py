import torch
from torch import nn
from .bert import BertTextEncoder
from .hybrid_layer import HybridLearningEncoder
from .hyper_layer import HyperLearningDecoder
from .unimodalcontrastive import UniContrastiveLoss


class FMFN(nn.Module):
    def __init__(self, args):
        super(FMFN, self).__init__()

        self.data_missing = args['base']['data_missing']

        self.bertmodel = BertTextEncoder(use_finetune=True,
                                         transformers='bert',
                                         pretrained=args['model']['feature_extractor']['bert_pretrained'])
        # project t,a,v to common dim
        self.proj_t0 = nn.Linear(args['model']['feature_extractor']['input_dim'][0], args['model']['com_dim'])
        self.proj_a0 = nn.Linear(args['model']['feature_extractor']['input_dim'][1], args['model']['com_dim'])
        self.proj_v0 = nn.Linear(args['model']['feature_extractor']['input_dim'][2], args['model']['com_dim'])

        self.hybrid_encoder_layer = HybridLearningEncoder(seq_len=args['model']['hle']['input_length'],
                                                          neck_size=args['model']['hle']['neck_size'],
                                                          dim=args['model']['hle']['hidden_dim'],
                                                          depth=args['model']['hle']['depth'],
                                                          heads=args['model']['hle']['attn_heads'],
                                                          dropout=args['model']['hle']['dropout'],
                                                          emb_dropout=args['model']['hle']['dropout'])

        
        self.uni_con_ta = UniContrastiveLoss(
            dim=args['model']['ucl']['hidden_dim'],
            neck_size=args['model']['ucl']['neck_size'])
        
        self.uni_con_tv = UniContrastiveLoss(
            dim=args['model']['ucl']['hidden_dim'],
            neck_size=args['model']['ucl']['neck_size'])


        self.hyper_decoder_layer = HyperLearningDecoder(dim=args['model']['hld']['hidden_dim'],
                                                        depth=args['model']['hld']['depth'],
                                                        heads=args['model']['hld']['attn_heads'],
                                                        dropout=args['model']['hld']['dropout'],
                                                        emb_dropout = args['model']['hld']['dropout'])

        self.classify = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['out_dim'])


    def forward(self,complete_input,incomplete_input):

        if self.data_missing:
            x_visual, x_audio, x_text = incomplete_input
        else:
            x_visual, x_audio, x_text = complete_input


        x_text = self.bertmodel(x_text)

        x_text = self.proj_t0(x_text)
        x_audio = self.proj_a0(x_audio)
        x_visual = self.proj_v0(x_visual)

        mast_a = mask_v = None
        mask_t = None


        n_t, n_a, n_v  = self.hybrid_encoder_layer(x_text, x_audio, x_visual, mask_t, mast_a, mask_v)



        u_con = self.uni_con_ta(n_a,n_t) + self.uni_con_tv(n_v,n_t)


        n_m = self.hyper_decoder_layer(n_t, n_a, n_v)


        n_f = torch.mean(n_m,dim=1)

        output = self.classify(n_f)

        return {'sentiment_preds': output,
                'token_con': u_con
                }



def build_model(args):
    return FMFN(args)