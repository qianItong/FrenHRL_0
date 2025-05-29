import argparse

feature_dim = 1024 # 最终特征维度

'''
option的对应关系
0: follow
1: left
2: right
3: left_U
4: right_U
5: stop
6: back
7: invalid
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)

    # 输入维度
    parser.add_argument("--in_dim_vehicle", type=int, default=7, help="自车信息")
    parser.add_argument("--in_dim_agents", type=int, default=8)
    parser.add_argument("--in_dim_static", type=int, default=6)
    parser.add_argument("--in_dim_lane", type=int, default=165, help="lane的维度")
    parser.add_argument("--in_dim_route", type=int, default=165, help="route的维度")
    parser.add_argument("--in_dim_reference_line", type=int, default=240, help="参考线的维度,120个二维点")

    parser.add_argument("--action_dim", type=int, default=32, help="动作的维度")

    parser.add_argument("--out_dim_lane", type=int, default=64)
    parser.add_argument("--out_dim_route", type=int, default=64)
    parser.add_argument("--feature_dim_agent", type=int, default=64)
    parser.add_argument("--out_dim_agent", type=int, default=128)
    parser.add_argument("--out_dim_static", type=int, default=32)

    parser.add_argument("--memory_len", type=int, default=8)
    parser.add_argument("--hidden_dim_T", type=int, default=128)
    parser.add_argument("--layer_num_T", type=int, default=2)
    parser.add_argument("--head_num_T", type=int, default=4)

    # 特征融合网络参数
    parser.add_argument("--hidden_dim_dense", type=int, default=128)
    parser.add_argument("--layer_num_dense", type=int, default=3)

    # 最终特征维度
    parser.add_argument("--feature_dim", type=int, default=feature_dim)

    # option嵌入维度
    parser.add_argument("--option_embedding_dim", type=int, default=32, help="option的嵌入维度")
    parser.add_argument("--option_num", type=int, default=7, help="option的数量,follow,left,right,left_U,right_U,stop,back")
    
    # 分类网络参数,用于将噪声库中的动作加权求和
    parser.add_argument("--hidden_dim_classify", type=int, default=128)
    parser.add_argument("--layer_num_classify", type=int, default=2)
    parser.add_argument("--out_dim_classify", type=int, default=32, help="分类的维度,也即分类数量")

    # 扩散模型参数
    parser.add_argument("--sde_hidden_dim", type=int, default=128)
    parser.add_argument("--sde_layer_num", type=int, default=3)
    parser.add_argument("--sde_T", type=int, default=5, help="时间长度")

    # 训练参数
    parser.add_argument('--save_every', default=25, type=int, help='每多少个epoch保存一次模型')
    parser.add_argument('--val_every', default=5, type=int, help='每多少个epoch评估一次模型')
    parser.add_argument('--total_epochs', default=300, type=int, help='总共训练多少个epoch')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    # 优化器参数，包括初始学习率,学习率衰减的epoch数,学习率衰减的gamma
    parser.add_argument('--rl_lr', default=0.001, type=float, help='初始learning rate')
    parser.add_argument('--diffusion_bc_lr', default=0.0002, type=float, help='扩散模型bc训练时的学习率')
    parser.add_argument('--diffusion_bc_step_size', default=20, type=int, help='学习率衰减的epoch数')
    parser.add_argument('--diffusion_bc_gamma', default=0.9, type=float, help='学习率衰减的gamma')
    

    # critic网络参数
    parser.add_argument("--critic_hidden_dim", type=int, default=128)
    parser.add_argument("--critic_layer_num", type=int, default=3)


    return parser.parse_args()
