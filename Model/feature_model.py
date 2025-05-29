import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=8):
        super(TransformerModel, self).__init__()

        # Transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = self.embedding(x)
        # 创建填充掩码
        src_key_padding_mask = self.create_padding_mask(x)
        # # Transformer编码器处理
        transformer_out = self.transformer_encoder(x,src_key_padding_mask = src_key_padding_mask)  # (batch_size, seq_len, hidden_size)
        last_out = self.select_last_out(transformer_out, src_key_padding_mask)
        # 通过全连接层得到最终输出
        output = self.fc(last_out)
        return output

    def create_padding_mask(self, x):
        # 创建一个掩码，指示输入的填充位置（全为零）
        mask = (x.sum(dim=-1) == 0).to(torch.bool)  # 假设填充为0
        mask[:, 0] = False  # 强制保留第一个位置，如果序列全是0，会导致掩码全为True，训练时会出错
        '''
            实际上，对于障碍物信息而言，对数据集进行了处理，有障碍物和无障碍物的数据分开训练，因此不会出现全为0的情况
            而对于记忆信息，考虑的是障碍物的历史信息，如果记忆中一直没有障碍物，就会全0，此时需要保留第一个位置
        '''
        return mask

    def select_last_out(self, transformer_out, src_key_padding_mask):
    # 反转mask，True表示有效位置
        reversed_mask = ~src_key_padding_mask  # (batch_size, seq_len)

        # 沿序列维度反转，找到最后一个有效位置
        reversed_mask_flipped = reversed_mask.flip(dims=[1])
        last_indices = reversed_mask_flipped.int().argmax(dim=1)
        last_indices = (reversed_mask.shape[1] - 1) - last_indices

        # 提取对应位置的输出
        last_out = transformer_out[torch.arange(transformer_out.size(0)), last_indices]
        return last_out
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num):
        super(MLP, self).__init__()
        self.layer_num = layer_num
        self.total_dim = input_dim
        self.mlp = nn.Sequential()
        self.mlp.add_module('input', nn.Linear(self.total_dim, hidden_dim))
        for i in range(layer_num):
            self.mlp.add_module(f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.mlp.add_module(f'activation_{i}', nn.Tanh())
        self.mlp.add_module('output', nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.mlp(x)

class MLP_RELU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 layer_num, norm=False, norm_layers=None, dropout=0.0):
        """
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层宽度
        :param output_dim: 输出维度
        :param layer_num: 隐藏层层数（Linear 数量）
        :param norm: 是否使用 LayerNorm
        :param norm_layers: 仅在前几层加 LayerNorm（int），如果为 None 则全部加
        :param dropout: Dropout 概率
        """
        super(MLP_RELU, self).__init__()
        self.mlp = nn.Sequential()

        # 输入层
        self.mlp.add_module('input', nn.Linear(input_dim, hidden_dim))
        if norm and (norm_layers > 0):
            self.mlp.add_module('input_norm', nn.LayerNorm(hidden_dim))
        self.mlp.add_module('input_act', nn.GELU())
        if dropout > 0:
            self.mlp.add_module('input_dropout', nn.Dropout(dropout))

        # 中间层
        for i in range(layer_num):
            self.mlp.add_module(f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim))
            if norm and (i < norm_layers-1):
                self.mlp.add_module(f'norm_{i}', nn.LayerNorm(hidden_dim))
            self.mlp.add_module(f'act_{i}', nn.ReLU())
            if dropout > 0:
                self.mlp.add_module(f'dropout_{i}', nn.Dropout(dropout))

        # 输出层
        self.mlp.add_module('output', nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.mlp(x)

class Memory_Module(nn.Module):
    def __init__(self, obstacle_feature_moudle, input_dim, hidden_dim, output_dim, layer_num, head_num):
        super(Memory_Module, self).__init__()
        self.obstacle_feature_moudle = obstacle_feature_moudle
        self.memory_feature_moudle = TransformerModel(
            input_size=input_dim,
            hidden_size=hidden_dim,
            output_size=output_dim,
            num_layers=layer_num,
            num_heads=head_num
        )
        
    def forward(self, memory):
        '''
        memory: (batch_size, seq_len_1, seq_len_2, input_dim)
        使用障碍物特征提取网络提取每一步记忆的特征，再用记忆特征网络提取整体特征
        '''
        batch_size, seq_len_1, seq_len_2, input_dim = memory.shape
        memory = memory.reshape(-1, seq_len_2, input_dim)   # (batch_size*seq_len_1, seq_len_2, input_dim)
        obstacle_feature = self.obstacle_feature_moudle(memory) # (batch_size*seq_len_1, feature_dim)
        obstacle_feature = obstacle_feature.reshape(batch_size, seq_len_1, -1)  # (batch_size, seq_len_1, feature_dim)
        memory_feature = self.memory_feature_moudle(obstacle_feature)
        return memory_feature

class Feature(nn.Module):
    def __init__(self, args, gpu_id=0):
        super(Feature, self).__init__()
        self.args = args
        self.T_lane = TransformerModel(
            input_size=self.args.in_dim_lane,
            hidden_size=self.args.hidden_dim_T,
            output_size=self.args.out_dim_lane,
            num_layers=self.args.layer_num_T,
            num_heads=self.args.head_num_T,
        )

        self.T_route = TransformerModel(
            input_size=self.args.in_dim_lane,
            hidden_size=self.args.hidden_dim_T,
            output_size=self.args.out_dim_route,
            num_layers=self.args.layer_num_T,
            num_heads=self.args.head_num_T,
        )

        self.T_static = TransformerModel(
            input_size=self.args.in_dim_static,
            hidden_size=self.args.hidden_dim_T,
            output_size=self.args.out_dim_static,
            num_layers=self.args.layer_num_T,
            num_heads=self.args.head_num_T,
        )
        self.T_memory = Memory_Module(
            obstacle_feature_moudle=TransformerModel(
                input_size=self.args.in_dim_agents,
                hidden_size=self.args.hidden_dim_T,
                output_size=self.args.feature_dim_agent,
                num_layers=self.args.layer_num_T,
                num_heads=self.args.head_num_T,
            ),
            input_dim=self.args.feature_dim_agent,
            hidden_dim=self.args.hidden_dim_T,
            output_dim=self.args.out_dim_agent,
            layer_num=self.args.layer_num_T,
            head_num=self.args.head_num_T,
        )
        self.reference_Mlp = MLP(
            input_dim=self.args.in_dim_reference_line,
            hidden_dim=64,
            output_dim=self.args.out_dim_route,
            layer_num=1,
        )
        self.dense_layer = MLP(
            input_dim=args.in_dim_vehicle + args.out_dim_lane + args.out_dim_agent +2*args.out_dim_route +args.out_dim_static,
            hidden_dim=args.hidden_dim_dense,
            output_dim=args.feature_dim,
            layer_num=args.layer_num_dense,
        )
        self.gpu_id = gpu_id
        
    def forward(self, data):
        ego = data['ego_current_state'].to(self.gpu_id)
        lanes = data['lanes'].to(self.gpu_id)
        route_lanes = data['route_lanes'].to(self.gpu_id)
        static_objects = data['static_objects'].to(self.gpu_id)
        agents_memory = data['neighbor_agents_past'].to(self.gpu_id)
        reference_line = data['reference_line'].to(self.gpu_id)
        reference_line = self.reference_Mlp(reference_line)
        lane_feature = self.T_lane(lanes)
        route_feature = self.T_route(route_lanes)
        static_feature = self.T_static(static_objects)
        memory_feature = self.T_memory(agents_memory)


        dense_input = torch.cat([ego, lane_feature, route_feature, memory_feature, static_feature, reference_line], dim=-1)
        feature = self.dense_layer(dense_input)

        return feature