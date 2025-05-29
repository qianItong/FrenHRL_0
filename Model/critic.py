import torch
import torch.nn as nn
from Model.feature_model import MLP_RELU

class OptionCritic(nn.Module):
    def __init__(self, args):
        super(OptionCritic, self).__init__()
        self.args = args
        self.option_embedding = nn.Embedding(args.option_num, args.option_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(args.feature_dim + 2 * args.option_embedding_dim, args.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.critic_hidden_dim, int(args.critic_hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(args.critic_hidden_dim/2), 1),
        )

    def forward(self, last_option, state, option):
        '''
        state: (batch_size, feature_dim)
        option: (batch_size, 1)
        '''
        embed_option = self.option_embedding(option)
        last_embed_option = self.option_embedding(last_option)
        state = torch.cat([last_embed_option, state, embed_option], dim=-1)
        return self.mlp(state)
    
class ActionCritic(nn.Module):
    def __init__(self, args, hidden_dim, layer_num):
        super(ActionCritic, self).__init__()
        self.args = args
        self.option_embedding = nn.Embedding(args.option_num, args.option_embedding_dim)
        self.mlp = MLP_RELU(
            input_dim=args.feature_dim + args.option_embedding_dim + args.action_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            layer_num=layer_num,
            norm=False,
            norm_layers=0,
            dropout=0.2,
        )

    def forward(self, state, option, action):
        '''
        state: (batch_size, feature_dim)
        option: (batch_size, option_num)
        action: (batch_size, action_dim)
        '''
        if len(option.shape) == 2: # (batch_size, option_num)
            option0 = option.squeeze(1) # (batch_size, )
        else:
            option0 = option
        embed_option = self.option_embedding(option0)
        state = torch.cat([state, embed_option, action], dim=-1)
        return self.mlp(state)