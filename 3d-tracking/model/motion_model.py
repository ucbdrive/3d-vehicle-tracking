import torch
import torch.nn as nn

import utils.network_utils as nu


class LSTMKF(nn.Module):
    '''
    LSTM in Kalman Filter format
    '''

    def __init__(self, batch_size, feature_dim, hidden_size, num_layers,
                 loc_dim):
        super(LSTMKF, self).__init__()
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loc_dim = loc_dim

        self.P = torch.zeros([batch_size, loc_dim, loc_dim])

        self.pred2loc = nn.Linear(
            hidden_size,
            loc_dim,
        )

        self.Q_noise = nn.Linear(
            hidden_size,
            loc_dim,
        )

        self.R_noise = nn.Linear(
            hidden_size,
            loc_dim,
        )

        self.pred_lstm = nn.LSTM(
            input_size=loc_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.Q_lstm = nn.LSTM(
            input_size=loc_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.R_lstm = nn.LSTM(
            input_size=loc_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self._init_param()

    def init_hidden(self, device):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(
                device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(
                device))

    def predict(self, location, pd_hc, q_hc):
        '''
        Predict location at t+1 using updated location at t
        Input:
            location: (B x 3), location from previous update
            hc_0: (num_layers, B, hidden_size), tuple of hidden and cell
        Middle:
            embed: (1, B x feature_dim), location feature
            out: (1 x B x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, B, hidden_size), tuple of updated hidden, cell
            output_pred: (B x loc_dim), predicted location
        '''
        B = location.shape[0]

        # Embed feature to hidden_size
        embed = location.view(1, B, self.loc_dim)

        out, pd_hc_n = self.pred_lstm(embed, pd_hc)
        q_, q_hc_n = self.Q_lstm(embed, q_hc)

        Q = torch.exp(self.Q_noise(q_)).view(B, -1)

        Q_ = []
        for i in range(B):
            Q_.append(torch.diag(Q[i]))

        self.Q = torch.cat(Q_).view(B, self.loc_dim, self.loc_dim)
        self.P = self.P.to(self.Q) + self.Q

        output_pred = self.pred2loc(out).view(B, self.loc_dim) + location

        return output_pred, pd_hc_n, q_hc_n

    def refine(self, location, observation, r_hc):
        '''
        Refine predicted location using single frame estimation at t+1 
        Input:
            location: (B x 3), location from prediction
            observation: (B x 3), location from single frame estimation
            hc_0: (num_layers, B, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, B x feature_dim), predicted location feature
            obs_embed: (1, B x feature_dim), single frame location feature
            embed: (1, B x 2*feature_dim), location feature
            out: (1 x B x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, B, hidden_size), tuple of updated hidden, cell
            output_pred: (B x loc_dim), predicted location
        '''
        B = location.shape[0]

        # Embed feature to hidden_size
        y_ = observation - location
        embed = observation.view(1, B, self.loc_dim)

        r_, r_hc_n = self.R_lstm(embed, r_hc)
        R_ = torch.exp(self.R_noise(r_)).view(B, -1)

        P_ = self.P.detach().to(R_)

        R_l = []
        inv_S_ = []
        for i in range(B):
            R_l.append(torch.diag(R_[i]))
            inv_S_.append(torch.inverse(P_[i] + R_l[i]))
        R = torch.cat(R_l).view(B, self.loc_dim, self.loc_dim)
        inv_S = torch.cat(inv_S_).view(B, self.loc_dim, self.loc_dim)

        S = P_ + R

        K = torch.matmul(P_, inv_S)

        output_pred = location + torch.matmul(K,
                                              y_.view(B, self.loc_dim, 1)).view(
            B, self.loc_dim)

        I_KH = torch.eye(self.loc_dim).to(K) - K

        self.P = torch.matmul(I_KH, torch.matmul(P_, I_KH.transpose(1, 2))) \
                 + torch.matmul(K, torch.matmul(R, K.transpose(1, 2)))

        return output_pred, r_hc_n

    def _init_param(self):
        nu.init_module(self.Q_noise)
        nu.init_module(self.R_noise)
        nu.init_module(self.pred2loc)
        nu.init_lstm_module(self.pred_lstm)
        nu.init_lstm_module(self.Q_lstm)
        nu.init_lstm_module(self.R_lstm)


class LSTM(nn.Module):

    '''
    Estimating object location in world coordinates
    Prediction LSTM:
        Input: 5 frames velocity
        Output: Next frame location
    Updating LSTM:
        Input: predicted location and observed location
        Output: Refined location
    '''

    def __init__(self, batch_size, feature_dim, hidden_size, num_layers,
                 loc_dim):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loc_dim = loc_dim

        self.loc2feat = nn.Linear(
            loc_dim,
            feature_dim,
        )

        self.pred2loc = nn.Linear(
            hidden_size,
            loc_dim,
            bias=False,
        )

        self.vel2feat = nn.Linear(
            loc_dim,
            feature_dim,
        )

        self.pred_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            dropout=0.5,
            num_layers=num_layers,
        )

        self.refine_lstm = nn.LSTM(
            input_size=2 * feature_dim,
            hidden_size=hidden_size,
            dropout=0.5,
            num_layers=num_layers,
        )

        self._init_param()

    def init_hidden(self, device):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(
                device),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(
                device))

    def predict(self, velocity, location, hc_0):
        '''
        Predict location at t+1 using updated location at t
        Input:
            velocity: (num_seq, num_batch, loc_dim), location from previous update
            location: (num_batch, loc_dim), location from previous update
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            embed: (num_seq, num_batch x feature_dim), location feature
            out: (num_seq x num_batch x hidden_size), lstm output
            merge_feat: (num_batch x hidden_size), the predicted residual
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_seq, num_batch, _ = velocity.shape

        # Embed feature to hidden_size
        embed = self.vel2feat(velocity).view(num_seq, num_batch, self.feature_dim)

        out, (h_n, c_n) = self.pred_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out[-1]

        output_pred = self.pred2loc(merge_feat).view(num_batch, self.loc_dim) + location

        return output_pred, (h_n, c_n)

    def refine(self, location, observation, hc_0):
        '''
        Refine predicted location using single frame estimation at t+1 
        Input:
            location: (num_batch x 3), location from prediction
            observation: (num_batch x 3), location from single frame estimation
            hc_0: (num_layers, num_batch, hidden_size), tuple of hidden and cell
        Middle:
            loc_embed: (1, num_batch x feature_dim), predicted location feature
            obs_embed: (1, num_batch x feature_dim), single frame location feature
            embed: (1, num_batch x 2*feature_dim), location feature
            out: (1 x num_batch x hidden_size), lstm output
            merge_feat: same as out
        Output:
            hc_n: (num_layers, num_batch, hidden_size), tuple of updated hidden, cell
            output_pred: (num_batch x loc_dim), predicted location
        '''
        num_batch = location.shape[0]

        # Embed feature to hidden_size
        loc_embed = self.loc2feat(location).view(num_batch, self.feature_dim)
        obs_embed = self.loc2feat(observation).view(num_batch, self.feature_dim)
        embed = torch.cat([loc_embed, obs_embed], dim=1).view(1, num_batch,
                                                              2 *
                                                              self.feature_dim)

        out, (h_n, c_n) = self.refine_lstm(embed, hc_0)

        # Merge embed feature with output
        # merge_feat = h_n + embed
        merge_feat = out

        output_pred = self.pred2loc(merge_feat).view(num_batch,
                                                     self.loc_dim) + observation

        return output_pred, (h_n, c_n)

    def _init_param(self):
        nu.init_module(self.loc2feat)
        nu.init_module(self.vel2feat)
        nu.init_module(self.pred2loc)
        nu.init_lstm_module(self.pred_lstm)
        nu.init_lstm_module(self.refine_lstm)


