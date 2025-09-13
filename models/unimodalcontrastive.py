import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UniContrastiveLoss(nn.Module):
    def __init__(self,dim, neck_size):
        super().__init__()
        # for fine-grained constrast weights
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(dim), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(neck_size), requires_grad=True) # num_words equal to shared token
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(neck_size), requires_grad=True) #num_frames
        self.logit_scale = nn.Parameter(torch.ones([]))
    def forward(self, audvis, text):
        """
        audvis:  all tokens of audio or visual       # [bs, num_tokens, dim]
        text:  all tokens of text       # [bs, num_tokens, dim]
        """
        # normalize to unit sphere
        audvis_output = audvis / audvis.norm(dim=-1, keepdim=True)
        text_output = text / text.norm(dim=-1, keepdim=True)

        audvis_text_logits = self.logit_scale * self._attenion_over_fine_grained_sim_matrix(audvis_output, text_output)
        nce_loss = self.nce(audvis_text_logits)

        return nce_loss
    def nce(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=1) # dim=-1
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss
    def _attenion_over_fine_grained_sim_matrix(self, frame_features, word_features):
        bs_frame, num_frames, dim_frame = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape

        fine_grained_sim_scores = torch.matmul(torch.matmul(frame_features.reshape(-1, dim_frame), self.local_mat_weight), word_features.reshape(-1, dim_text).t()).reshape(bs_frame, num_frames, bs_text, num_words)  # [bs_frame, num_frames, bs_text, num_words]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores / 1e-2, dim=-1),
                                                   self.word_mat_weight) * fine_grained_sim_scores, dim=-1)

        frame2word_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=1).permute(0,2,1), self.frame_mat_weight).permute(0,2,1) * word_level_logit, dim=1)  # [bs_frame, bs_text]


        return frame2word_logits

