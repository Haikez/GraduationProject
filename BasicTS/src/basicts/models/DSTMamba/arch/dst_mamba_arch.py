import torch
import torch.nn as nn
import torch.nn.functional as F

from .RevIN import RevIN
from .SeriesDec import Temporal_Decomposition
from .SeriesMix import MultiScaleTrendMixing
from .Embed import DataEmbedding
from .MambaEnc import Encoder, EncoderLayer
from .SCITrendDownsample import SCITrendDownsample
from .IDMambaSeasonal import IDMambaSeasonal
from .STIDGCNSeasonal import STIDGCNSeasonal
from ..config import DSTMambaConfig


class DSTMamba(nn.Module):
    def __init__(self, config: DSTMambaConfig):
        super(DSTMamba, self).__init__()

        # Keep the source model's parameters while accepting a BasicTS config.
        model_args = config

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.d_model = model_args['d_model']

        self.use_norm = model_args['use_norm']
        #self.norm_type = model_args['norm_type']
        self.emb_dropout = model_args['emb_dropout']
        self.decom_type = model_args['decom_type']
        self.std_kernel = model_args['std_kernel']

        self.rank = model_args['rank']
        self.node_dim = model_args['node_dim']

        # TODO: Change the sequential mode of Mamba
        # self.mamba_mode = model_args['mamba_mode']
        # assert self.mamba_mode in ['bi-directional', 'uni-directional']

        self.e_layers = model_args['e_layers']
        self.d_state = model_args['d_state']
        self.d_conv = model_args['d_conv']
        self.expand = model_args['expand']

        self.d_ff = model_args['d_ff']
        self.ffn_dropout = model_args['ffn_dropout']
        self.ffn_activation = model_args['ffn_activation']

        self.ds_type = model_args['ds_type']
        if self.ds_type not in ('max', 'avg', 'conv'):
            raise ValueError("ds_type must be 'max', 'avg', or 'conv'")
        self.ds_layers = model_args['ds_layers']
        self.ds_window = model_args['ds_window']
        self.trend_downsampling = model_args.get('trend_downsampling', 'source')
        self.sci_hidden_channels = model_args.get('sci_hidden_channels', 8)
        self.sci_kernel_size = model_args.get('sci_kernel_size', 3)
        self.sci_dropout = model_args.get('sci_dropout', 0.0)
        self.seasonal_encoder = model_args.get('seasonal_encoder', 'dst_mamba')
        self.idmamba_channels = model_args.get('idmamba_channels', 48)
        self.idmamba_granularity = model_args.get('idmamba_granularity', 288)
        self.idmamba_dropout = model_args.get('idmamba_dropout', 0.1)
        self.stidgcn_channels = model_args.get('stidgcn_channels', 48)
        self.stidgcn_granularity = model_args.get('stidgcn_granularity', 288)
        self.stidgcn_dropout = model_args.get('stidgcn_dropout', 0.1)

        self.initial_tre_w = model_args['initial_tre_w']
        self.mamba_backend = model_args.get('mamba_backend', 'official')

        if self.d_model <= self.node_dim or self.rank < 1:
            raise ValueError("d_model must exceed node_dim and rank must be positive")
        if self.ds_window < 1 or self.ds_layers < 1:
            raise ValueError("ds_window and ds_layers must be positive")
        if self.trend_downsampling not in ('source', 'avg', 'sci', 'sci_low'):
            raise ValueError("trend_downsampling must be 'source', 'avg', 'sci', or 'sci_low'")
        if self.seasonal_encoder not in ('dst_mamba', 'idmamba', 'stidgcn'):
            raise ValueError("seasonal_encoder must be 'dst_mamba', 'idmamba', or 'stidgcn'")
        if (self.idmamba_channels < 1 or self.idmamba_granularity < 1
                or not 0.0 <= self.idmamba_dropout < 1.0):
            raise ValueError("ID-Mamba channels/granularity must be positive and dropout in [0, 1)")
        if self.seasonal_encoder == 'idmamba' and self.history_seq_len % 4:
            raise ValueError("ID-Mamba seasonal tree requires history length divisible by 4")
        if (self.stidgcn_channels < 1 or self.stidgcn_granularity < 1
                or not 0.0 <= self.stidgcn_dropout < 1.0):
            raise ValueError("STIDGCN channels/granularity must be positive and dropout in [0, 1)")
        if self.seasonal_encoder == 'stidgcn' and self.history_seq_len % 4:
            raise ValueError("STIDGCN seasonal tree requires history length divisible by 4")
        if self.trend_downsampling in ('sci', 'sci_low'):
            if self.ds_window != 2:
                raise ValueError("SCI trend downsampling requires ds_window=2")
            if self.history_seq_len % (2 ** self.ds_layers):
                raise ValueError("SCI trend downsampling requires history length divisible by 2**ds_layers")
        if self.history_seq_len // self.ds_window ** self.ds_layers < 1:
            raise ValueError("Downsampling would reduce history to zero steps")
        if self.decom_type != 'STD':
            raise ValueError("Only the source model's STD decomposition is supported")
        if self.std_kernel < 1 or self.std_kernel % 2 != 1:
            raise ValueError("std_kernel must be a positive odd number")

        self.build()


    def build(self):

        Mamba = None
        if self.seasonal_encoder != 'stidgcn':
            if self.mamba_backend == 'official':
                try:
                    from mamba_ssm import Mamba
                except ImportError as exc:
                    raise ImportError(
                        "The official backend requires mamba-ssm; use "
                        "mamba_backend='torch' on a Mac or CPU."
                    ) from exc
            elif self.mamba_backend == 'torch':
                from .torch_mamba import TorchMamba as Mamba
            else:
                raise ValueError("mamba_backend must be 'official' or 'torch'")

        self.revin_layer = RevIN(num_features=self.num_channels)

        if self.decom_type == 'STD':
            self.decom = Temporal_Decomposition(self.std_kernel)

        if self.seasonal_encoder == 'dst_mamba':
            embed_dim = self.d_model - self.node_dim
            self.embedding = DataEmbedding(self.history_seq_len, embed_dim, self.emb_dropout)

            self.adapter = nn.Parameter(torch.empty(self.num_channels, embed_dim, self.rank)) # [N, E, r]
            nn.init.xavier_uniform_(self.adapter)
            self.lora = nn.Linear(self.rank, self.node_dim, bias=False)

            self.encoder = Encoder(
                [
                    EncoderLayer(
                        ssm     = Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                        ssm_r   = Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                        d_model = self.d_model,
                        d_ff    = self.d_ff,
                        dropout = self.ffn_dropout,
                        activation = self.ffn_activation
                    ) for layer in range(self.e_layers)
                ],
                # TODO: Test the effectiveness of _RMSNorm_
                norm_layer = nn.LayerNorm(self.d_model)
            )

            self.projector = nn.Linear(self.d_model, self.future_seq_len, bias=True)

        elif self.seasonal_encoder == 'idmamba':
            self.seasonal_model = IDMambaSeasonal(
                history_len=self.history_seq_len,
                horizon=self.future_seq_len,
                num_nodes=self.num_channels,
                channels=self.idmamba_channels,
                granularity=self.idmamba_granularity,
                mamba_cls=Mamba,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.idmamba_dropout,
            )
        else:
            self.seasonal_model = STIDGCNSeasonal(
                history_len=self.history_seq_len,
                horizon=self.future_seq_len,
                num_nodes=self.num_channels,
                channels=self.stidgcn_channels,
                granularity=self.stidgcn_granularity,
                dropout=self.stidgcn_dropout,
            )

        if self.trend_downsampling in ('sci', 'sci_low'):
            pass  # Build SCI modules after the shared layers for paired-seed comparisons.
        elif self.trend_downsampling == 'avg' or self.ds_type == 'avg':
            self.down_pool = nn.AvgPool1d(self.ds_window)
        elif self.ds_type == 'max':
            self.down_pool = nn.MaxPool1d(self.ds_window, return_indices=False)
        elif self.ds_type == 'conv':
            padding = 1
            self.down_pool = nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                                       kernel_size=3, padding=padding, stride=self.ds_window,
                                       padding_mode='circular', bias=False)

        scale_lengths = [self.history_seq_len]
        for _ in range(self.ds_layers):
            length = scale_lengths[-1]
            scale_lengths.append(
                (length + self.ds_window - 1) // self.ds_window
                if self.trend_downsampling == 'source' and self.ds_type == 'conv'
                else length // self.ds_window
            )
        self.ms_mixing = MultiScaleTrendMixing(
            self.history_seq_len, self.future_seq_len, self.num_channels,
            self.ds_layers, self.ds_window, scale_lengths=scale_lengths
        )

        self.linear_mappings = nn.ModuleList(
            [
                nn.Linear(length, self.future_seq_len) for length in scale_lengths
            ]
        )
        self.tre_w = nn.Parameter(torch.FloatTensor([self.initial_tre_w]*self.num_channels), requires_grad=True)

        if self.trend_downsampling in ('sci', 'sci_low'):
            # Preserve the RNG stream used for data shuffling in paired runs.
            with torch.random.fork_rng(devices=[]):
                self.sci_downsamplers = nn.ModuleList([
                    SCITrendDownsample(
                        hidden_channels=self.sci_hidden_channels,
                        kernel_size=self.sci_kernel_size,
                        dropout=self.sci_dropout,
                    ) for _ in range(self.ds_layers)
                ])
                if self.trend_downsampling == 'sci':
                    self.sci_detail_mappings = nn.ModuleList([
                        nn.Linear(length, self.future_seq_len, bias=False)
                        for length in scale_lengths[1:]
                    ])
                    for mapping in self.sci_detail_mappings:
                        nn.init.zeros_(mapping.weight)


    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor = None) -> torch.Tensor:

        # BasicTS supplies [batch, history, nodes]; timestamps are unused.
        if inputs.ndim != 3 or inputs.shape[1:] != (self.history_seq_len, self.num_channels):
            raise ValueError(
                f"inputs must have shape [batch, {self.history_seq_len}, {self.num_channels}]"
            )
        x_in = inputs

        if self.use_norm:
            x_in = self.revin_layer(x_in, mode='norm')

        x_sea, x_trend = self.decom(x_in)

        if self.seasonal_encoder == 'dst_mamba':
            # Embedding: [B, T, N] -> [B, N, E]
            x_emb = self.embedding(x_sea)

            # Add spatial embeddings
            adapter = F.relu(self.lora(self.adapter)) # [N, E-D, r] -> [N, E-D, D]
            adaptation = torch.einsum('bne,ned->bnd', x_emb, adapter)
            x_emb = torch.cat([x_emb, adaptation], dim=-1)  # [B, N, E]

            # TODO: Add Graph or reduce sequential bias

            # Encoder: [B, N, E] -> [B, N, E]
            enc_out = self.encoder(x_emb)

            x_sea_out = self.projector(enc_out).permute(0, 2, 1) # (B, N, d_model) -> (B, N, T) -> (B, T, N)

        else:
            x_sea_out = self.seasonal_model(x_sea, inputs_timestamps)

        # The source route downsamples raw inputs, then decomposes each scale.
        # The new routes decompose once, then downsample only the trend series.
        if self.trend_downsampling == 'source':
            ms_list = [x_in]
            x_ms = x_in.permute(0, 2, 1)
            for _ in range(self.ds_layers):
                x_ms = self.down_pool(x_ms)
                ms_list.append(x_ms.permute(0, 2, 1))
            ms_trend_list = [self.decom(x)[1] for x in ms_list]
        else:
            ms_trend_list = [x_trend]
            current = x_trend
            detail_list = []
            for layer in range(self.ds_layers):
                if self.trend_downsampling in ('sci', 'sci_low'):
                    current, detail = self.sci_downsamplers[layer](current)
                    if self.trend_downsampling == 'sci':
                        detail_list.append(detail)
                else:
                    current = self.down_pool(current.permute(0, 2, 1)).permute(0, 2, 1)
                ms_trend_list.append(current)

        ms_trend_list = self.ms_mixing(ms_trend_list)

        # multi-scale mappings
        out_trend_list = []
        for i, trend in zip(range(len(ms_trend_list)), ms_trend_list):
            trend_out = self.linear_mappings[i](trend.permute(0, 2, 1)).permute(0, 2, 1)
            out_trend_list.append(trend_out)

        x_tre_out = torch.stack(out_trend_list, dim=-1).sum(-1)
        if self.trend_downsampling == 'sci':
            detail_forecasts = [
                mapping(detail.permute(0, 2, 1)).permute(0, 2, 1)
                for mapping, detail in zip(self.sci_detail_mappings, detail_list)
            ]
            x_tre_out = x_tre_out + torch.stack(detail_forecasts, dim=-1).sum(-1)

        # Weighted Sum
        prediction = x_sea_out + self.tre_w * x_tre_out
        if self.use_norm:
            prediction = self.revin_layer(prediction, mode='denorm')

        return prediction
