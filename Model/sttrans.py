import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def get_torch_trans(heads=16, layers=8, channels=512):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=1024, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, nheads, dropout=0.6):
        super(ResidualBlock, self).__init__()
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=4, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=4, channels=channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward_time(self, y, base_shape):
        B, channel, L = base_shape
        if L == 1:
            return y
        y = y.permute(0, 2, 1)
        y = self.time_layer(y)
        y = y.permute(0, 2, 1)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, L = base_shape
        if channel == 1:
            return y
        y = y.permute(0, 2, 1)
        y = self.feature_layer(y)
        y = y.permute(0, 2, 1)
        return y

    def forward(self, x, cond_info):
        B, channel, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, L)
        y = x
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)
        _, cond_dim = cond_info.shape[:2]
        cond_info = cond_info.view(B, cond_dim, L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return self.relu((x + residual) / math.sqrt(2.0)), skip

class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, x):
        return x

class Postprocess(nn.Module):
    def __init__(self):
        super(Postprocess, self).__init__()

    def forward(self, x):
        return x

class ST_Trans(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=8, nhead=16, dim_feedforward=512, dropout=0.6):
        super(ST_Trans, self).__init__()
        self.num_joints = input_dim // 3
        self.side_dim = dim_feedforward
        self.preprocess = Preprocess()
        self.postprocess = Postprocess()
        self.conv_projection1 = Conv1d_with_init(self.num_joints * 2, self.num_joints * 4, kernel_size=3)
        self.conv_projection2 = Conv1d_with_init(self.num_joints * 4, self.num_joints * 8, kernel_size=3)
        self.conv_projection3 = Conv1d_with_init(self.num_joints * 8, self.num_joints * 4, kernel_size=3)
        self.conv_projection4 = Conv1d_with_init(self.num_joints * 4, self.num_joints * 2, kernel_size=3)
        self.embedding = nn.Linear(self.num_joints * 2, dim_feedforward)
        self.transformer_encoder = get_torch_trans(heads=nhead, layers=num_layers, channels=dim_feedforward)
        self.fc = nn.Linear(dim_feedforward, num_classes)
        self.residual_blocks = nn.ModuleList([ResidualBlock(self.side_dim, dim_feedforward, nhead, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.preprocess(x)
        B, L, _ = x.shape
        viability_scores = x[:, :, 2::3]
        x_coords = x[:, :, 0::3]
        y_coords = x[:, :, 1::3]
        coords = torch.cat([x_coords, y_coords], dim=2)
        coords = coords.permute(0, 2, 1)
        coords = self.conv_projection1(coords)
        coords = self.conv_projection2(coords)
        coords = self.conv_projection3(coords)
        coords = self.conv_projection4(coords)
        coords = coords.permute(0, 2, 1)
        x = self.embedding(coords)
        viability_weights = viability_scores.unsqueeze(-1)
        viability_weights = viability_weights.repeat(1, 1, 1, x.size(-1) // viability_scores.size(-1))
        viability_weights = viability_weights.view(B, L, -1)[:, :, :x.size(-1)]
        if viability_weights.shape[2] != x.shape[2]:
            padding = x.shape[2] - viability_weights.shape[2]
            if padding > 0:
                viability_weights = F.pad(viability_weights, (0, padding))
            else:
                viability_weights = viability_weights[:, :, :x.shape[2]]
        x = x * viability_weights
        side_info = torch.ones(B, self.side_dim, L, device=x.device)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        for layer in self.residual_blocks:
            x, _ = layer(x, side_info)
        x = x.mean(dim=2)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.postprocess(x)
        return x

if __name__ == "__main__":
    model = ST_Trans(input_dim=36, num_classes=4)
    print(model)




