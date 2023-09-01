import torch
import torch.nn.functional as fnn
import einops as eop
from torch import nn
from torch.distributions import Normal, Independent


class EncoderModel(nn.Module):
    def __init__(self, params):
        super(EncoderModel, self).__init__()
        self.params = params
        self.encoder_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
        )

    def forward(self, obs):
        encoded_obs = self.encoder_net(obs)
        encoded_obs = eop.rearrange(encoded_obs, 'b c h w -> b (c h w)')
        return encoded_obs


class RepresentationModel(nn.Module):
    def __init__(self, params):
        super(RepresentationModel, self).__init__()
        self.params = params
        self.repr_net = FeedForwardNet(
            input_dim=self.params['h_dim']+self.params['feat_dim'],
            output_dim=2*self.params['z_dim'],
            hidden_dim=self.params['h_dim'],
            n_layers=self.params['n_ff_layers']
        )

    def forward(self, h_state, encoded_obs):
        concat_input = torch.concat([h_state, encoded_obs], dim=1)
        mu, pre_std = torch.chunk(self.repr_net(concat_input), chunks=2, dim=1)
        std = fnn.softplus(pre_std + 0.55) + self.params['min_std']
        dist_posterior = Independent(Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)
        return dist_posterior


class RecurrentModel(nn.Module):
    def __init__(self, params, action_dim):
        super(RecurrentModel, self).__init__()
        self.params = params
        self.gru_net = nn.GRUCell(input_size=self.params['z_dim']+action_dim, hidden_size=self.params['h_dim'])

    def forward(self, h_state, z_state, action):
        gru_input = torch.concat([z_state, action], dim=1)
        next_h_state = self.gru_net(gru_input, h_state)
        return next_h_state


class TransitionModel(nn.Module):
    def __init__(self, params):
        super(TransitionModel, self).__init__()
        self.params = params
        self.transition_net = FeedForwardNet(
            input_dim=self.params['h_dim'],
            output_dim=2*self.params['z_dim'],
            hidden_dim=self.params['h_dim'],
            n_layers=self.params['n_ff_layers']
        )

    def forward(self, h_state):
        mu, pre_std = torch.chunk(self.transition_net(h_state), chunks=2, dim=1)
        std = fnn.softplus(pre_std + 0.55) + self.params['min_std']
        dist_prior = Independent(Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)
        return dist_prior


class DecoderModel(nn.Module):
    def __init__(self, params):
        super(DecoderModel, self).__init__()
        self.params = params
        self.fc_net = nn.Linear(in_features=self.params['h_dim']+self.params['z_dim'], out_features=1024)
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=6),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, stride=2, kernel_size=6)
        )

    def forward(self, h_state, z_state):
        concat_input = torch.concat([h_state, z_state], dim=-1)
        fc_output = self.fc_net(concat_input)
        reshaped_input = fc_output.view(-1, 256, 2, 2)
        mu_obs = self.decoder_net(reshaped_input)
        dist_obs = Independent(Normal(loc=mu_obs, scale=1.0), reinterpreted_batch_ndims=3)
        return dist_obs


class RewardModel(nn.Module):
    def __init__(self, params):
        super(RewardModel, self).__init__()
        self.params = params
        self.reward_net = FeedForwardNet(
            input_dim=params['h_dim']+params['z_dim'],
            output_dim=1,
            hidden_dim=params['h_dim'],
            n_layers=self.params['n_ff_layers']
        )

    def forward(self, h_state, z_state):
        concat_input = torch.concat([h_state, z_state], dim=-1)
        mu_reward = self.reward_net(concat_input)
        dist_reward = Independent(Normal(loc=mu_reward, scale=1.0), reinterpreted_batch_ndims=1)
        return dist_reward


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(FeedForwardNet, self).__init__()
        self.to_hidden = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.ReLU()
            ) for _ in range(n_layers-1)
        ])
        self.from_hidden = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, x):
        to_hidden = self.to_hidden(x)
        hidden = self.hidden(to_hidden)
        return self.from_hidden(hidden)
