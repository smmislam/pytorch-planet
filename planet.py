# References
#       PlaNet Paper: https://arxiv.org/pdf/1811.04551
#
#       Implementation:
#           1. Danijar's repo: [https://github.com/danijar/planet]
#           1. Jaesik's repo: [https://github.com/jsikyoon/dreamer-torch]
#           2. Kaito's repo: [https://github.com/cross32768/PlaNet_PyTorch]

import torch
from torch import nn
from torch.distributions import kl_divergence
from networks import EncoderModel, RepresentationModel, RecurrentModel, TransitionModel, DecoderModel, RewardModel
from utils import get_device, get_dtype


class Planet(nn.Module):
    def __init__(self, params, action_dim):
        super(Planet, self).__init__()
        self.params = params
        self.d_type = get_dtype(self.params['fp_precision'])
        self.device = get_device(self.params['device'])
        self.action_dim = action_dim
        self.rnn_model = RecurrentModel(params=self.params, action_dim=self.action_dim)
        self.obs_encoder = EncoderModel(params=self.params)
        self.repr_model = RepresentationModel(params=self.params)
        self.transition_model = TransitionModel(params=self.params)
        self.decoder_model = DecoderModel(params=self.params)
        self.reward_model = RewardModel(params=self.params)

    def __repr__(self):
        return 'PlaNet'

    def get_init_h_state(self, batch_size):
        return torch.zeros((batch_size, self.params['h_dim']), dtype=self.d_type, device=self.device)

    def forward(self, sampled_episodes):
        dist_predicted = {'prior': list(), 'posterior': list(), 'recon_obs': list(), 'reward': list()}
        h_state = self.get_init_h_state(batch_size=self.params['batch_size'])
        for time_stamp in range(self.params['chunk_length']):
            input_obs = sampled_episodes['obs'][time_stamp]
            noisy_input_obs = (1/pow(2, self.params['pixel_bit']))*torch.randn_like(input_obs) + input_obs
            action = sampled_episodes['action'][time_stamp]

            encoded_obs = self.obs_encoder(noisy_input_obs)
            z_prior = self.transition_model(h_state)
            z_posterior = self.repr_model(h_state, encoded_obs)

            z_state = z_posterior.rsample()

            dist_recon_obs = self.decoder_model(h_state, z_state)
            dist_reward = self.reward_model(h_state, z_state)
            h_state = self.rnn_model(h_state, z_state, action)

            dist_predicted['prior'].append(z_prior)
            dist_predicted['posterior'].append(z_posterior)
            dist_predicted['recon_obs'].append(dist_recon_obs)
            dist_predicted['reward'].append(dist_reward)
        return dist_predicted

    def compute_loss(self, target, dist_predicted):
        sampled_reconstructed_obs = torch.stack([dist_recon_obs.rsample() for dist_recon_obs in dist_predicted['recon_obs']])
        sampled_reward = torch.stack([dist_reward.rsample() for dist_reward in dist_predicted['reward']])
        # Individual loss terms
        recon_loss = ((target['obs'] - sampled_reconstructed_obs) ** 2).mean(dim=0).mean(dim=0).sum()
        kl_loss = torch.stack(
            [kl_divergence(p=dist_posterior, q=dist_prior) for dist_prior, dist_posterior in
             zip(dist_predicted['prior'], dist_predicted['posterior'])]
        )
        kl_loss = (torch.maximum(kl_loss, torch.tensor([self.params['free_nats']])[0])).mean()
        reward_prediction_loss = ((target['reward'] - sampled_reward) ** 2).mean()
        # Net loss term
        net_loss = recon_loss + kl_loss + reward_prediction_loss
        return net_loss, (recon_loss.item(), kl_loss.item(), reward_prediction_loss.item())

