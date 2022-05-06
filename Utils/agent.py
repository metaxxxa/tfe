
import torch
import torch.nn as nn

class AgentNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.RNN = args.RNN
        self.CONVOLUTIONAL_INPUT = args.CONVOLUTIONAL_INPUT
        if self.RNN:
            if args.CONVOLUTIONAL_INPUT:
                outconv1 = int((args.ENV_SIZE - args.DILATION*(args.KERNEL_SIZE-1) + 2*args.PADDING - 1)/args.STRIDE + 1)
                outmaxPool = int((outconv1 + 2*args.PADDING_POOL - args.DILATION*(args.KERNEL_SIZE_POOL - 1) -1)/args.STRIDE + 1) #see full formula on pytorch's documentation website
                outconv2 = int((outmaxPool - args.DILATION2*(args.KERNEL_SIZE2-1) + 2*args.PADDING2 - 1)/args.STRIDE2 + 1)
                outmaxPool2 = int((outconv2 + 2*args.PADDING_POOL2 - args.DILATION2*(args.KERNEL_SIZE_POOL2 - 1) -1)/args.STRIDE2 + 1) 
                self.conv_layer = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=args.CONV_OUT_CHANNELS, kernel_size=(args.KERNEL_SIZE, args.KERNEL_SIZE), padding=args.PADDING, stride = args.STRIDE, dilation = args.DILATION),
                    nn.ReLU(),
                    nn.MaxPool2d(args.KERNEL_SIZE_POOL, args.STRIDE, dilation=args.DILATION, padding=args.PADDING_POOL),
                    nn.Conv2d(in_channels=args.CONV_OUT_CHANNELS, out_channels=args.CONV_OUT_CHANNELS2, kernel_size=(args.KERNEL_SIZE2, args.KERNEL_SIZE2), padding=args.PADDING2, stride = args.STRIDE2, dilation = args.DILATION2),
                    nn.ReLU(),
                    nn.MaxPool2d(args.KERNEL_SIZE_POOL2, args.STRIDE2, dilation=args.DILATION2, padding=args.PADDING_POOL2),
                    nn.Flatten(0,-1)
                ).to(args.device)
                self.mlp1 = nn.Linear(args.CONV_OUT_CHANNELS2*outmaxPool2**2, args.dim_L1_agents_net, device=args.device)
                self.gru = nn.GRUCell(args.dim_L1_agents_net,args.dim_L2_agents_net, device=args.device)
                self.mlp2 = nn.Linear(args.dim_L2_agents_net, args.n_actions, device=args.device)
                self.relu = nn.ReLU()
            else:
                self.mlp1 = nn.Linear(args.nb_inputs_agent, args.dim_L1_agents_net, device=args.device)
                self.gru = nn.GRUCell(args.dim_L1_agents_net,args.dim_L2_agents_net, device=args.device)
                self.mlp2 = nn.Linear(args.dim_L2_agents_net, args.n_actions, device=args.device)
                self.relu = nn.ReLU()
        else:
            if args.CONVOLUTIONAL_INPUT:
                outconv1 = int((args.ENV_SIZE - args.DILATION*(args.KERNEL_SIZE-1) + 2*args.PADDING - 1)/args.STRIDE + 1)
                outmaxPool = int((outconv1 + 2*args.PADDING_POOL - args.DILATION*(args.KERNEL_SIZE_POOL - 1) -1)/args.STRIDE + 1) #see full formula on pytorch's documentation website
                outconv2 = int((outmaxPool - args.DILATION2*(args.KERNEL_SIZE2-1) + 2*args.PADDING2 - 1)/args.STRIDE2 + 1)
                outmaxPool2 = int((outconv2 + 2*args.PADDING_POOL2 - args.DILATION2*(args.KERNEL_SIZE_POOL2 - 1) -1)/args.STRIDE2 + 1) 
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=args.CONV_OUT_CHANNELS, kernel_size=(args.KERNEL_SIZE, args.KERNEL_SIZE), padding=args.PADDING, stride = args.STRIDE, dilation = args.DILATION),
                    nn.ReLU(),
                    nn.MaxPool2d(args.KERNEL_SIZE_POOL, args.STRIDE, dilation=args.DILATION, padding=args.PADDING_POOL),
                    nn.Conv2d(in_channels=args.CONV_OUT_CHANNELS, out_channels=args.CONV_OUT_CHANNELS2, kernel_size=(args.KERNEL_SIZE2, args.KERNEL_SIZE2), padding=args.PADDING2, stride = args.STRIDE2, dilation = args.DILATION2),
                    nn.ReLU(),
                    nn.MaxPool2d(args.KERNEL_SIZE_POOL2, args.STRIDE2, dilation=args.DILATION2, padding=args.PADDING_POOL2),
                    
                    nn.Flatten(0,-1),
                    nn.Linear(args.CONV_OUT_CHANNELS2*outmaxPool2**2, args.hidden_layer1_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_layer1_dim,args.hidden_layer2_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_layer2_dim, args.n_actions)
                ).to(args.device)

            else:
                self.net = nn.Sequential(
                    nn.Linear(args.observations_dim , args.hidden_layer1_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_layer1_dim,args.hidden_layer2_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_layer2_dim, args.n_actions)
                ).to(args.device)
            

    def forward(self, obs_t, hidden_state_t):
        if self.RNN:
            if self.CONVOLUTIONAL_INPUT:
                input = self.conv_layer(obs_t)
            else:
                input = obs_t
            in_gru = self.relu(self.mlp1(input))
            hidden_next = self.gru(in_gru.unsqueeze(0), hidden_state_t)
            q_values = self.mlp2(hidden_next)
            return q_values, hidden_next
        else:
            return self.net(obs_t), None

if __name__ == '__main__':
    class Args:
        CONVOLUTIONAL_INPUT = False
        RNN = False
        observations_dim  = 3
        hidden_layer1_dim = 5
        hidden_layer2_dim = 7
        n_actions = 11
        device = 'cpu'

    agentnet = AgentNet(Args)