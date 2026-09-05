import torch
import torch.nn as nn
import torch.nn.functional as F

from aad.models.proposed.CAA import InceptionBottleneck
from aad.models.proposed.mlp import MLP
from aad.models.proposed.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows

class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )

        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        x = self.gcn3d(x)

        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        return x

class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size=3,
                 window_stride=1,
                 window_dilation=1):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
        ])

    def forward(self, x):
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        return out_sum




class BeaST(nn.Module):
    def __init__(self, num_class, num_point_left, num_point_right, num_g3d_scales, num_gcn_scales, in_channels, out_channels,c1,graph_left, graph_right):
        super(BeaST, self).__init__()

        self.left_model = Model(num_class, num_point_left, num_g3d_scales, num_gcn_scales, in_channels, out_channels,c1,graph_left)
        self.right_model = Model(num_class, num_point_right, num_g3d_scales, num_gcn_scales, in_channels, out_channels,c1,graph_right)
        c2 = 32
        c3 = c2*2

        self.fc = nn.Linear(72, num_class)  # 融合两个脑半球的输出，注意通道数翻倍
        out_channels1 = 8
        self.sigmoid = nn.Sigmoid()
        self.diff_brain1 = nn.Sequential(
            nn.Conv2d(64, out_channels1, kernel_size=1),
            nn.BatchNorm2d(out_channels1),
            nn.LeakyReLU(0.1),
        )
        num_jpts = graph_right.shape[-1]
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa_coarse_to_fine_block_3_to_3 = nn.Conv1d(c3+out_channels1, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa_coarse_to_fine_block_3_to_3.weight)  #
        nn.init.constant_(self.conv_sa_coarse_to_fine_block_3_to_3.bias, 0)



    def forward(self, left_x, right_x):
        N = left_x.size(0)

        left_out = self.left_model(left_x)
        right_out = self.right_model(right_x)



        diff_brain_l = left_out - right_out
        x1_1 = torch.cat((self.diff_brain1(diff_brain_l),left_out),dim = 1)

        diff_brain_r = right_out - left_out
        x2_1 = torch.cat((self.diff_brain1(diff_brain_r),right_out),dim = 1)
        L = x1_1
        R = x2_1
        batch_size,features,T, electrodes = L.size()

        L_pooled = torch.mean(L, dim=2, keepdim=True)  # (B, F, 1, E)
        R_pooled = torch.mean(R, dim=2, keepdim=True)  # (B, F, 1, E)

        attention_weights_R_to_L = F.softmax(torch.bmm(R_pooled.view(batch_size, features, electrodes),
                                                L_pooled.view(batch_size, electrodes, features)), dim=-1)  # (B, E, F)

        attention_weights_L_to_R = F.softmax(torch.bmm(L_pooled.view(batch_size, features, electrodes),
                                                R_pooled.view(batch_size, electrodes, features)), dim=-1)  # (B, E, F)

        R_weighted = torch.bmm(attention_weights_R_to_L, L_pooled.view(batch_size, features, electrodes))  # (B, E, F)
        L_weighted = torch.bmm(attention_weights_L_to_R, R_pooled.view(batch_size, features, electrodes))  # (B, E, F)

        R_weighted = R_weighted.view(batch_size, features, 1, electrodes)  # (B, E, 1, F)
        L_weighted = L_weighted.view(batch_size, features, 1, electrodes)  # (B, E, 1, F)

        x1_1 = L + L_weighted  # (B, F, T, E) + (B, E, 1, F)
        x2_1 = R + R_weighted  # (B, F, T, E) + (B, E, 1, F)



        se_spatial1 = x2_1.mean(-2)  # N' C' V'
        se1_spatial1 = self.sigmoid(self.conv_sa_coarse_to_fine_block_3_to_3(se_spatial1))
        x1_1 = x1_1 * se1_spatial1.unsqueeze(-2) + x1_1 # N, C, T, V * N, C, 1, V

        se_spatial = x1_1.mean(-2)  # N' C' V'
        se1_spatial = self.sigmoid(self.conv_sa_coarse_to_fine_block_3_to_3(se_spatial))
        x2_1 = x2_1 * se1_spatial.unsqueeze(-2) + x2_1 # N, C, T, V * N, C, 1, V

        out_channels_l = x1_1.size(1)
        out1 = x1_1.view(N, out_channels_l, -1)
        out1 = out1.mean(2)

        final_out_l = self.fc(out1)  # 最终全连接层分类


        out_channels_r = x2_1.size(1)
        out2 = x2_1.view(N, out_channels_r, -1)
        out2 = out2.mean(2)

        final_out_r = self.fc(out2)  # 最终全连接层分类

        return final_out_l, final_out_r


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,  # Adjusted to 64 for the EEG electrodes
                 num_g3d_scales,
                 num_gcn_scales,
                 in_channels,
                 out_channels,
                 c1,
                 graph # We'll pass the precomputed adjacency matrix
                 ):  # Number of time samples per trial
        super(Model, self).__init__()

        A_binary = graph
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)

        in1 = int(out_channels//2 + 8*(out_channels/4))
        c2 = c1*2
        c3 = c2*2

        self.gcn3d1 = MultiWindow_MS_G3D(in1, c1, A_binary, num_g3d_scales)
        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales,window_stride=1)
        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales,window_stride=2)


        self.Inseption_block1 = InceptionBottleneck(c1)
        self.Inseption_block2 = InceptionBottleneck(c2)
        self.Inseption_block3 = InceptionBottleneck(c3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in1, c1, (2,1), padding=0, stride=(1,1)),
            nn.BatchNorm2d(c1)
        )
        self.input_map = nn.Sequential(
            nn.Conv2d(1, out_channels//2, 1),
            nn.BatchNorm2d(out_channels//2),
            nn.LeakyReLU(0.2),
        )
        self.diff_map1 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )
        self.diff_map2 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )
        self.diff_map3 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )
        self.diff_map4 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )
        self.diff_map5 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )
        self.diff_map6 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )
        self.diff_map7 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )
        self.diff_map8 = nn.Sequential(
            nn.Conv2d(1, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.LeakyReLU(0.2),
        )



    def forward(self, x):

        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N,  V * C, T)

        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0,2,3,1).contiguous()

        dif1 = x[:, :, 1:] - x[:, :, 0:-1]
        dif1 = torch.cat([dif1.new(N, C, 1, V).zero_(), dif1], dim=-2)
        dif2 = x[:, :, 2:] - x[:, :, 0:-2]
        dif2 = torch.cat([dif2.new(N, C, 2, V).zero_(), dif2], dim=-2)

        dif3 = x[:, :, 3:] - x[:, :, 0:-3]
        dif3 = torch.cat([dif3.new(N, C, 3, V).zero_(), dif3], dim=-2)

        dif7 = x[:, :, 4:] - x[:, :, :-4]
        dif7 = torch.cat([dif7.new(N, C, 4, V).zero_(), dif7], dim=-2)


        dif4 = x[:, :, :-1] - x[:, :, 1:]
        dif4 = torch.cat([dif4, dif4.new(N, C, 1, V).zero_()], dim=-2)

        dif5 = x[:, :, :-2] - x[:, :, 2:]
        dif5 = torch.cat([dif5, dif5.new(N, C, 2, V).zero_()], dim=-2)

        dif6 = x[:, :, :-3] - x[:, :, 3:]
        dif6 = torch.cat([dif6, dif6.new(N, C, 3, V).zero_()], dim=-2)

        dif8 = x[:, :, :-4] - x[:, :, 4:]
        dif8 = torch.cat([dif8, dif8.new(N, C, 4, V).zero_()], dim=-2)

        x = torch.cat((self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2), self.diff_map3(dif3), self.diff_map7(dif7),self.diff_map4(dif4), self.diff_map5(dif5), self.diff_map6(dif6), self.diff_map8(dif8)),dim = 1)



        x = F.relu(self.gcn3d1(x), inplace=True)
        x = F.relu(self.Inseption_block1(x), inplace=True)

        x = F.relu(self.gcn3d2(x), inplace=True)
        x = F.relu(self.Inseption_block2(x), inplace=True)

        x = F.relu(self.gcn3d3(x), inplace=True)
        x = F.relu(self.Inseption_block3(x), inplace=True)


        out = x
        return out
