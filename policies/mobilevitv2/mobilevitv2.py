import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import sys
sys.path.append('/home/djr/DISCOVERSE')
from policies.common.wrapper import TemporalEnsembling
from MobileViTv2_pytorch.MobileVit_v2_new import MobileViTv2_New
import numpy as np
import cv2

def get_sinusoid_encoding_table(n_position, d_hid):
    """生成正弦余弦位置编码表，返回形状 (1, n_position, d_hid)"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class MobileViTPolicy(nn.Module):
    def __init__(self, args_override):
        super(MobileViTPolicy, self).__init__()
        
        joint_num = args_override["joint_num"]  # 默认为7
        self.cam_num = len(args_override["camera_names"])  # 从配置中获取相机数量
        self.num_queries = args_override["num_queries"]  # 默认为25  # chunk_size
        self.lr = args_override["lr"]
        # 初始化 MobileViT 模型，输出形状为 (B, 512, 14, 14)
        self.mobilevit = MobileViTv2_New()
        self.hidden_features = self.mobilevit.model_conf_dict["output"]["out"]

        # TODO:14*14还没想好从哪获取
        # 使用正弦位置嵌入（sinusoidal positional embedding），形状 (1, 14*14, 512)
        self.register_buffer('pos_embed', get_sinusoid_encoding_table(14 * 14, self.hidden_features))
        
        # 关节角度特征处理：将12维关节角转换为512维
        self.joint_fc = nn.Linear(joint_num, self.hidden_features)
        # 对 self.joint_fc 层的权重进行初始化，使用的是截断正态分布（truncated normal distribution）。
        nn.init.trunc_normal_(self.joint_fc.weight, std=0.02)
        
        # 特征融合后的处理（这里使用简单的逐元素加法融合）
        self.fusion_fc = nn.Linear(self.hidden_features, self.hidden_features)
        
        # 动作序列编码层
        self.encoder_action_proj = nn.Linear(joint_num, self.hidden_features) 

        # 输出层，预测12个关节角
        self.output_fc = nn.Linear(self.hidden_features, joint_num)
        
        # 根据 cam_num 构造 MLP 的输入维度：
        # 动作嵌入维度：hidden_features  --这个没有必要继续输入了
        # 关节特征维度：hidden_features  
        # 所有相机图像特征拼接：cam_num * hidden_features  
        mlp_input_dim = (1 + self.cam_num) * self.hidden_features
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, joint_num)
        )
        
        self._args = args_override
        # 保存参数
        self._args = args_override
        
        # 设置优化器（后面在configure_optimizers中配置）
        self.optimizer = None
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        self.temporal_ensembler = None
        try:
            if args_override["temporal_agg"]:
                self.temporal_ensembler = TemporalEnsembling(
                    args_override["chunk_size"],
                    args_override["action_dim"],
                    args_override["max_timesteps"],
                )
        except Exception as e:
            print(e)
            print(
                "The above Exception can be ignored when training instead of evaluating."
            )
        print(f"Initialized MobileViTPolicy for predicting {joint_num} joint angles")

    def reset(self):
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def __call__(self, joint_angles, image:torch.Tensor, actions=None, is_pad=None):
        """
            joint_angles: (B, joint_num) 当前关节角
            image: (B, num_cam, 3, H, W) 多视角图像（或单视角时 num_cam=1）
            actions: (B, seq, joint_num) 历史动作序列，需要进行嵌入（此处会截取前 25 步）
            target_actions: (B, seq, joint_num) 目标动作序列，用于计算损失（训练时提供）
            is_pad: (B, seq) 是否为填充部分，用于计算损失（训练时提供）
        """  
        # 如果输入 image 为单张图片 (3, H, W)，则加上 batch 维度
        if image.ndim == 4:
            image = image.unsqueeze(0) # 加上batch维度
        # print(image.shape)# e.g. torch.Size([4, 2, 3, 448, 448])
        # 转换为单通道，反正是灰度图
        # TODO：理论上推理直接是单通道灰度图
        image = image.mean(dim=2, keepdim=True)
        # 应用图像预处理
        # image = self.normalize(image)
        image.div_(255.0) 
        

        if actions is not None:
            
            # 处理动作序列特征
            # actions = actions[:, : self.model.num_queries]
            actions = actions[:, : self.num_queries] #chunk_size # shape =torch.Size([4, 25, 7]) (bs, seq, joint_num)
            assert is_pad is not None, "is_pad should not be None"
            is_pad = is_pad[:, : self.num_queries]
            # action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)  # 无需Embed输入，因为我们不是VAE模型
            # action_embed shape = torch.Size([4, 25, 512])
            # 处理关节角特征，将 (B, 12) 投影到 (B, 512)
            joint_features = self.joint_fc(joint_angles)
            
            # 多视角情况： image shape 为 (B, cam_num, 3, H, W)
            cam_features = []
            for i in range(self.cam_num):
                view = image[:, i, :, :, :]  # (B, 1, H, W)
                feat = self.mobilevit.forward(view)  # (B, hidden_features, 28, 28)
                #feat shape =torch.Size([4, 512, 14, 14])
                # 扁平化空间维度，并添加正弦位置嵌入
                feat_seq = feat.flatten(2).transpose(1, 2) + self.pos_embed  # (B, 196, hidden_features)
                # 全局池化得到图像全局特征
                global_feat = feat_seq.mean(dim=1)  # (B, hidden_features)
                cam_features.append(global_feat)
            # 将所有相机的全局特征按特征维度拼接，得到 (B, cam_num * hidden_features)
            global_feats_cat = torch.cat(cam_features, dim=-1)
            # 将关节特征和图像全局特征扩展到动作序列的时间步维度
            # seq_len = action_embed.size(1)
            seq_len = self.num_queries
            joint_features_exp = joint_features.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq, hidden_features)
            global_feats_cat_exp = global_feats_cat.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq, cam_num*hidden_features)

            # 拼接动作嵌入、关节特征和所有相机图像特征，得到融合特征 (B, seq, (1+cam_num)*hidden_features)
            fused_features = torch.cat([joint_features_exp, global_feats_cat_exp], dim=-1)
            
            # 输入 MLP，得到预测的动作序列 (B, seq, joint_num)  
            predicted_actions = self.mlp(fused_features)

            all_l1 = F.l1_loss(actions, predicted_actions, reduction="none")

            l1_loss = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict = {'loss': l1_loss}
            return loss_dict
        else:  # 推理模式  ---待检验debug
            # no action, sample from prior
            
            # 处理关节角特征
            joint_features = self.joint_fc(joint_angles)
            
            # 处理图像（多视角），得到各个视角的全局图像特征
            cam_features = []
            for i in range(self.cam_num):
                view = image[:, i, :, :, :]  # (B, 3, H, W)
                feat = self.mobilevit.forward(view)  # (B, hidden_features, 28, 28)
                feat_seq = feat.flatten(2).transpose(1, 2) + self.pos_embed  # (B, 196, hidden_features)
                global_feat = feat_seq.mean(dim=1)  # (B, hidden_features)
                cam_features.append(global_feat)
            global_feats_cat = torch.cat(cam_features, dim=-1)  # (B, cam_num*hidden_features)
            
            # 将关节特征和图像全局特征扩展到时间步维度
            seq_len = self.num_queries
            joint_features_exp = joint_features.unsqueeze(1).expand(-1, seq_len, -1)
            global_feats_cat_exp = global_feats_cat.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 融合各部分特征
            fused_features = torch.cat([joint_features_exp, global_feats_cat_exp], dim=-1)
            predicted_actions = self.mlp(fused_features)
            
            # 如果存在时序集成器，则更新第一个时间步的输出
            if self.temporal_ensembler is not None:
                a_hat_one = self.temporal_ensembler.update(predicted_actions)
                predicted_actions[0][0] = a_hat_one
            return predicted_actions

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer
