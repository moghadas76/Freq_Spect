import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim
        # simple MLP on scalar t in [0,1]
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, dim)
        )

    def forward(self, t: torch.Tensor):
        # t: [B, 1, 1] or [B, 1]
        if t.dim() == 3:
            t = t.view(t.size(0), 1)
        return self.net(t)  # [B, dim]


class Model(nn.Module):
    """
    Flow_SpectFlow
    - Same reconstruction path as Real_SpectFlow to produce full-length xy (seq_len + pred_len)
    - Additional flow head flow(x_t, t) to regress the velocity field for flow matching
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.cut_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # frequency upsampler (same as Real_SpectFlow but with explicit real/imag mixing)
        if self.individual:
            self.freq_upsampler_real = nn.ModuleList()
            self.freq_upsampler_imag = nn.ModuleList()
            for _ in range(self.channels):
                self.freq_upsampler_real.append(nn.Linear(self.cut_freq, int(self.cut_freq * self.length_ratio)))
                self.freq_upsampler_imag.append(nn.Linear(self.cut_freq, int(self.cut_freq * self.length_ratio)))
            print("..................", self.freq_upsampler_real[-1])
        else:
            self.freq_upsampler_real = nn.Linear(self.cut_freq, int(self.cut_freq * self.length_ratio))
            self.freq_upsampler_imag = nn.Linear(self.cut_freq, int(self.cut_freq * self.length_ratio))

        # flow head (built dynamically on first use to match actual channels)
        self.flow_time_dim = getattr(configs, 'flow_time_dim', 16)
        self.flow_hidden_multiplier = float(getattr(configs, 'flow_hidden_multiplier', 2.0))
        self.t_embed = TimeEmbed(self.flow_time_dim)
        self.flow_net: nn.Module | None = None
        self._flow_in_dim = None
        self._flow_out_dim = None

        # Channel-wise Multi-Head Attention (after RIN)
        self.chan_attn_dim = int(getattr(configs, 'chan_attn_dim', 64))
        self.chan_attn_heads = int(getattr(configs, 'chan_attn_heads', 4))
        # ensure divisibility
        if self.chan_attn_dim % self.chan_attn_heads != 0:
            self.chan_attn_dim = self.chan_attn_heads * ((self.chan_attn_dim + self.chan_attn_heads - 1) // self.chan_attn_heads)
        self.chan_qkv_in = nn.Linear(1, self.chan_attn_dim)
        self.chan_attn = nn.MultiheadAttention(self.chan_attn_dim, self.chan_attn_heads, batch_first=True)
        self.chan_out = nn.Linear(self.chan_attn_dim, 1)
        self.chan_attn_dropout = nn.Dropout(getattr(configs, 'dropout', 0.0))
        self.chan_attn_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        # RIN (RevIN-like per instance normalization)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # Channel-attention over input channels (per time step)
        # reshape to [B*T, C, 1] -> project to embeddings -> MHA across C tokens -> project back to scalar per channel
        B, T, C = x.shape
        z = self.chan_qkv_in(x.view(B * T, C, 1))                # [B*T, C, D]
        z_attn, _ = self.chan_attn(z, z, z)                      # [B*T, C, D]
        z_attn = self.chan_attn_dropout(z_attn)
        x_attn = self.chan_out(z_attn).view(B, T, C)             # [B, T, C]
        x = x + self.chan_attn_alpha * x_attn                    # residual mix

        # frequency-domain low-pass and upsample
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx = torch.view_as_real(low_specx[:, 0:self.cut_freq, :])
        low_specx_real = low_specx[:, :, :, 0]
        low_specx_imag = low_specx[:, :, :, 1]

        if self.individual:
            # not commonly used in this repo; keep simple sum of real/imag paths
            up_real = []
            up_imag = []
            for i in range(self.channels):
                r = self.freq_upsampler_real[i](low_specx_real[:, :, i].permute(0, 1)).permute(0, 1)
                im = self.freq_upsampler_imag[i](low_specx_imag[:, :, i].permute(0, 1)).permute(0, 1)
                up_real.append(r)
                up_imag.append(im)
            low_specxy_real = torch.stack(up_real, dim=2)
            low_specxy_imag = torch.stack(up_imag, dim=2)
        else:
            low_specxy_real = self.freq_upsampler_real(low_specx_real.permute(0, 2, 1)).permute(0, 2, 1) - \
                               self.freq_upsampler_imag(low_specx_imag.permute(0, 2, 1)).permute(0, 2, 1)
            low_specxy_imag = self.freq_upsampler_real(low_specx_imag.permute(0, 2, 1)).permute(0, 2, 1) + \
                               self.freq_upsampler_imag(low_specx_real.permute(0, 2, 1)).permute(0, 2, 1)

        # pad to target FFT length (seq_len+pred_len)
        target_fft_len = int((self.seq_len + self.pred_len) / 2 + 1)
        low_specxy_R = torch.zeros([low_specxy_real.size(0), target_fft_len, low_specxy_real.size(2)],
                                   dtype=low_specxy_real.dtype, device=low_specxy_real.device)
        low_specxy_I = torch.zeros([low_specxy_imag.size(0), target_fft_len, low_specxy_imag.size(2)],
                                   dtype=low_specxy_imag.dtype, device=low_specxy_imag.device)
        low_specxy_R[:, 0:low_specxy_real.size(1), :] = low_specxy_real
        low_specxy_I[:, 0:low_specxy_imag.size(1), :] = low_specxy_imag

        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # compensate length change

        xy = low_xy * torch.sqrt(x_var) + x_mean
        return xy, 0

    def _ensure_flow_net(self, c_in: int, device):
        in_dim = c_in + self.flow_time_dim
        out_dim = c_in
        if (self.flow_net is None) or (self._flow_in_dim != in_dim) or (self._flow_out_dim != out_dim):
            hidden = max(64, int(c_in * self.flow_hidden_multiplier))
            self.flow_net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            ).to(device)
            self._flow_in_dim = in_dim
            self._flow_out_dim = out_dim

    def flow(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        Predict velocity field v(x_t, t).
        x_t: [B, T, C] or [B, C, T]
        t: [B, 1, 1] or [B, 1]
        returns v_pred with shape [B, T, C]
        """
        if x_t.dim() != 3:
            raise ValueError(f"x_t must be 3D [B, T, C] or [B, C, T], got {x_t.shape}")
        # correct possible [B, C, T] layout
        if x_t.size(1) == self.channels and x_t.size(2) != self.channels:
            x_t = x_t.transpose(1, 2)  # -> [B, T, C]

        B, T, C = x_t.shape
        # build flow head for actual channel count
        self._ensure_flow_net(C, x_t.device)

        t_emb = self.t_embed(t)  # [B, D]
        t_emb = t_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, D]
        inp = torch.cat([x_t, t_emb], dim=-1)  # [B, T, C+D]
        v = self.flow_net(inp)  # [B, T, C]
        return v
