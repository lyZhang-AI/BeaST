
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EEGCrossAttention1(nn.Module):
    def __init__(self, dropout=0.1):
        super(EEGCrossAttention1, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, left_brain_data, right_brain_data):

        batch_size, num_features, num_time_steps, num_left_electrodes = left_brain_data.shape
        _, _, _, num_right_electrodes = right_brain_data.shape

        left_query = left_brain_data.view(batch_size, num_features * num_left_electrodes, num_time_steps)
        right_key = right_brain_data.view(batch_size, num_features * num_right_electrodes, num_time_steps)

        d = left_query.shape[-2]

        scores_lr = torch.bmm(left_query.transpose(1, 2), right_key) / math.sqrt(d)
        attention_weights_lr = self.dropout(F.softmax(scores_lr, dim=-1))
        output_lr = torch.bmm(attention_weights_lr, right_key.transpose(1, 2))

        output_lr = output_lr.view(batch_size,  num_features, num_time_steps,num_right_electrodes)


        return output_lr


class EEGCrossAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(EEGCrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, left_brain_data, right_brain_data):

        batch_size, num_features, num_time_steps, num_left_electrodes = left_brain_data.shape
        _, _, _, num_right_electrodes = right_brain_data.shape


        left_query = left_brain_data.view(batch_size, num_features * num_left_electrodes, num_time_steps)
        right_key = right_brain_data.view(batch_size, num_features * num_right_electrodes, num_time_steps)

        d = left_query.shape[-2]


        scores_lr = torch.bmm(left_query.transpose(1, 2), right_key) / math.sqrt(d)
        attention_weights_lr = self.dropout(F.softmax(scores_lr, dim=-1))
        output_lr = torch.bmm(attention_weights_lr, right_key.transpose(1, 2))


        output_lr = output_lr.view(batch_size,  num_features, num_time_steps,num_right_electrodes)

        right_query = right_brain_data.view(batch_size, num_features * num_right_electrodes, num_time_steps)
        left_key = left_brain_data.view(batch_size, num_features * num_left_electrodes, num_time_steps)

        scores_rl = torch.bmm(right_query.transpose(1, 2), left_key) / math.sqrt(d)
        attention_weights_rl = self.dropout(F.softmax(scores_rl, dim=-1))
        output_rl = torch.bmm(attention_weights_rl, left_key.transpose(1, 2))


        output_rl = output_rl.view(batch_size,  num_features, num_time_steps, num_left_electrodes)

        output = torch.cat((output_lr, output_rl), dim=-1)


        return output
