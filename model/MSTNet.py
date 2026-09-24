import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== Normalization ======================
def min_max_normalize(x: Tensor) -> Tensor:
    """Perform min‑max normalization for each sample
    Args:
        x: input tensor (B, L, C)
    Returns:
        normalized tensor (B, L, C)
    """
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    x_range = x_max - x_min
    x_range[x_range < 1e-8] = 1e-8  # avoid division‑by‑zero
    x_norm = (x - x_min) / x_range
    return x_norm


def compute_moment_stats(window_data: Tensor) -> Tensor:
    """Compute high‑order moment features within each sliding window: mean, std, skewness, kurtosis
    Args:
        window_data: (N_win, B, C, w)
    Returns:
        4‑dimensional statistical features (N_win, B, C, 4)
    """
    w = window_data.size(-1)
    # mean value
    mu = window_data.mean(dim=-1, keepdim=True)
    # standard deviation (unbiased estimation, denominator w‑1)
    std = window_data.std(dim=-1, keepdim=True, unbiased=True)
    std[std < 1e-8] = 1e-8
    # standardized signal
    x_norm = (window_data - mu) / std
    # skewness
    skewness = (x_norm ** 3).sum(dim=-1, keepdim=True) / (w - 1)
    # kurtosis
    kurtosis = (x_norm ** 4).sum(dim=-1, keepdim=True) / (w - 1)
    # concatenate four statistical features
    stats = torch.cat([mu, std, skewness, kurtosis], dim=-1)
    return stats


def compute_spectral_features_per_frame(amp_spec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute three frame‑wise spectral features: spectral centroid, spectral flatness, spectral bandwidth
    Aligned with Figure 2 in the manuscript, calculated frame‑by‑frame
    Args:
        amp_spec: amplitude spectrum (B, F, T)
    Returns:
        Cf, Ff, Bf: frame‑level features (B, T). Sample‑level features are obtained by averaging along time dimension
    """
    B, F, T = amp_spec.shape
    f_idx = torch.arange(1, F + 1, device=amp_spec.device, dtype=amp_spec.dtype).view(1, F, 1)
    sum_A = amp_spec.sum(dim=1, keepdim=True)  # sum over frequency dimension
    sum_A[sum_A < 1e-8] = 1e-8
    # 1. spectral centroid, frame‑wise
    sum_fA = (f_idx * amp_spec).sum(dim=1, keepdim=True)
    cf = sum_fA / sum_A  # (B,1,T)
    # 2. spectral flatness (log transformation to prevent numerical underflow), frame‑wise
    log_A = torch.log(amp_spec + 1e-8)
    geo_mean = torch.exp(log_A.mean(dim=1, keepdim=True))
    arith_mean = amp_spec.mean(dim=1, keepdim=True)
    ff = geo_mean / (arith_mean + 1e-8)  # (B,1,T)
    # 3. spectral bandwidth, frame‑wise
    f_diff = (f_idx - cf) ** 2
    sum_fdiff_A = (f_diff * amp_spec).sum(dim=1, keepdim=True)
    bf = torch.sqrt(sum_fdiff_A / sum_A + 1e-8)  # (B,1,T)
    return cf.squeeze(1), ff.squeeze(1), bf.squeeze(1)  # each output: (B, T)


# ====================== MFM ======================
class MomentFeatureModule(nn.Module):
    def __init__(self, in_len: int = 10000, in_chan: int = 6):
        super().__init__()
        # fixed hyper‑parameters from manuscript
        self.win_size = 400
        self.overlap = 0.5
        self.stride = int(self.win_size * (1 - self.overlap))
        self.in_len = in_len
        self.in_chan = in_chan
        # calculate total number of sliding windows
        self.n_win = (self.in_len - self.win_size) // self.stride + 1
        self.feat_dim_per_chan = 4 * self.n_win
        # linear projection layers
        self.linear1 = nn.Linear(self.feat_dim_per_chan, 256)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256, 128)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: normalized input (B, L=10000, C=6)
        return: output of Moment Feature Module (B, 128)
        """
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        # perform sliding‑window segmentation
        x_win = x.unfold(dimension=-1, size=self.win_size, step=self.stride)
        x_win = x_win.permute(2, 0, 1, 3)  # (n_win, B, C, win_size)
        # compute moment statistics for each window
        win_stats = compute_moment_stats(x_win)
        win_stats = win_stats.permute(1, 2, 0, 3)  # (B, C, n_win, 4)
        feat_pre = win_stats.flatten(start_dim=2)   # (B, C, 4*n_win)
        # linear transformation
        feat_mid = self.relu(self.linear1(feat_pre))
        feat_mid = self.linear2(feat_mid)
        # global average pooling along channel dimension
        feat_mmt = feat_mid.mean(dim=1)
        return feat_mmt


# ====================== SFM ======================
class SpectrumFeatureModule(nn.Module):
    def __init__(self, in_len: int = 10000, in_chan: int = 6):
        super().__init__()
        self.stft_win = 256
        self.stft_stride = int(self.stft_win * 0.5)
        self.n_fft = 256
        self.hann_win = torch.hann_window(self.stft_win)
        # linear layers for spectral feature processing
        self.linear_amp = nn.Linear(129, 32)
        self.linear_spec = nn.Linear(3, 32)
        self.linear_fusion = nn.Linear(64, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: normalized input (B, L=10000, C=6)
        return: output of Spectrum Feature Module (B, 128)
        """
        B, L, C = x.shape
        feat_list = []
        # process each sensor channel independently
        for c in range(C):
            sig = x[:, :, c]  # (B, L)
            # short‑time Fourier transform
            spec_complex = torch.stft(
                sig,
                n_fft=self.n_fft,
                hop_length=self.stft_stride,
                win_length=self.stft_win,
                window=self.hann_win.to(sig.device),
                return_complex=True,
                normalized=False
            )
            amp_spec = torch.abs(spec_complex)  # (B, F, T)
            sum_amp = amp_spec.sum(dim=1, keepdim=True)
            sum_amp[sum_amp < 1e-8] = 1e-8
            weight = amp_spec / sum_amp
            amp_weighted = amp_spec * weight
            # global average of weighted amplitude spectrum along time axis
            amp_global = amp_weighted.mean(dim=-1)
            feat_amp = self.linear_amp(amp_global)
            # calculate three spectral features frame‑wise and average over time
            cf_frame, ff_frame, bf_frame = compute_spectral_features_per_frame(amp_spec)
            cf_sample = cf_frame.mean(dim=-1)
            ff_sample = ff_frame.mean(dim=-1)
            bf_sample = bf_frame.mean(dim=-1)
            spec_triple = torch.stack([cf_sample, ff_sample, bf_sample], dim=-1)
            feat_spec = self.relu(self.linear_spec(spec_triple))
            # concatenate amplitude‑related and statistical spectral features for single channel
            feat_concat = torch.cat([feat_amp, feat_spec], dim=-1)
            feat_list.append(feat_concat)
        # aggregate features across all channels
        feat_all = torch.stack(feat_list, dim=0).mean(dim=0)
        feat_freq = self.linear_fusion(feat_all)
        return feat_freq


# ====================== TFM ======================
class TemporalFeatureModule(nn.Module):
    def __init__(self, in_len: int = 10000, in_chan: int = 6):
        super().__init__()
        self.in_len = in_len
        self.in_chan = in_chan
        self.target_len = in_len // 8
        # fine‑scale branch
        self.conv_fine = nn.Conv1d(in_chan, in_chan, kernel_size=3, padding=1)
        self.bn_fine = nn.BatchNorm1d(in_chan)
        self.pool_fine = nn.MaxPool1d(kernel_size=2, stride=2)
        # medium‑scale branch
        self.conv_med = nn.Conv1d(in_chan, in_chan, kernel_size=15, padding=7)
        self.bn_med = nn.BatchNorm1d(in_chan)
        self.pool_med = nn.MaxPool1d(kernel_size=4, stride=4)
        # coarse‑scale branch
        self.conv_coarse = nn.Conv1d(in_chan, in_chan, kernel_size=31, padding=15)
        self.bn_coarse = nn.BatchNorm1d(in_chan)
        self.pool_coarse = nn.MaxPool1d(kernel_size=8, stride=8)
        # adaptive pooling to unify sequence length
        self.adap_pool = nn.AdaptiveAvgPool1d(self.target_len)
        # bidirectional GRU for temporal dependency modeling
        self.bi_gru = nn.GRU(
            input_size=in_chan * 3,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        # output projection
        self.linear_out = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: normalized input (B, L=10000, C=6)
        return: output of Temporal Feature Module (B, 128)
        """
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        f_fine = self.pool_fine(self.relu(self.bn_fine(self.conv_fine(x))))
        f_fine = self.adap_pool(f_fine)
        f_med = self.pool_med(self.relu(self.bn_med(self.conv_med(x))))
        f_med = self.adap_pool(f_med)
        f_coarse = self.pool_coarse(self.relu(self.bn_coarse(self.conv_coarse(x))))
        f_coarse = self.adap_pool(f_coarse)
        # concatenate multi‑scale temporal features
        f_concat = torch.cat([f_fine, f_med, f_coarse], dim=1)
        f_concat = f_concat.permute(0, 2, 1)  # (B, L/8, 3C)
        # bidirectional GRU forward pass
        gru_out, _ = self.bi_gru(f_concat)
        gru_global = gru_out.mean(dim=1)  # global pooling over time steps
        # feature dimension mapping
        feat_temp = self.linear_out(gru_global)
        return feat_temp


# ======================AMFM ======================
class AdaptiveMultimodalFusion(nn.Module):
    def __init__(self, feat_dim: int = 128, num_modal: int = 3, num_heads: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_modal = num_modal
        self.num_heads = num_heads
        # learnable weight for each modality
        self.modal_weight = nn.Parameter(torch.randn(num_modal))
        # multi‑head self‑attention for modality interaction
        self.mhsa = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        # layer normalization
        self.ln = nn.LayerNorm(feat_dim)

    def forward(self, feat_mmt: Tensor, feat_freq: Tensor, feat_temp: Tensor) -> Tensor:
        """
        Input: three modality‑specific features, each with shape (B, 128)
        Output: fused multimodal feature (B, 128)
        """
        # stack three modality features
        modal_stack = torch.stack([feat_mmt, feat_freq, feat_temp], dim=1)  # (B, 3, 128)
        # normalize learnable modality weights via softmax
        weight = F.softmax(self.modal_weight, dim=0).view(1, self.num_modal, 1)
        modal_weighted = modal_stack * weight
        # multi‑head self‑attention for cross‑modality interaction
        attn_out, _ = self.mhsa(modal_weighted, modal_weighted, modal_weighted)
        # global aggregation and layer normalization
        fuse_global = attn_out.mean(dim=1)
        fuse_feat = self.ln(fuse_global)
        return fuse_feat


# ======================MSTNet======================
class MSTNet(nn.Module):
    def __init__(self, in_len: int = 10000, in_channel: int = 6):
        super().__init__()
        # three parallel feature extraction branches
        self.mfm = MomentFeatureModule(in_len, in_channel)
        self.sfm = SpectrumFeatureModule(in_len, in_channel)
        self.tfm = TemporalFeatureModule(in_len, in_channel)
        # adaptive multimodal fusion module
        self.amfm = AdaptiveMultimodalFusion(feat_dim=128, num_modal=3, num_heads=4)
        # prediction head for water‑cut estimation, output range [0,1]
        self.pred_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # 1. apply min‑max normalization to raw input
        x_norm = min_max_normalize(x)
        # 2. extract features from three parallel branches
        feat_mmt = self.mfm(x_norm)
        feat_freq = self.sfm(x_norm)
        feat_temp = self.tfm(x_norm)
        # 3. adaptive multimodal feature fusion
        feat_fuse = self.amfm(feat_mmt, feat_freq, feat_temp)
        # 4. predict water‑cut value
        water_cut_pred = self.pred_head(feat_fuse)
        return water_cut_pred


# ====================== For Debug ======================
if __name__ == "__main__":
    # initialize model instance
    model = MSTNet().to(DEVICE)
    model.eval()
    # construct dummy input tensor (B=32, L=10000, C=6)
    test_input = torch.randn(32, 10000, 6).to(DEVICE)
    with torch.no_grad():
        output = model(test_input)
    print(f"Input shape: {test_input.shape}")    # expected: torch.Size([32, 10000, 6])
    print(f"Output shape: {output.shape}")       # expected: torch.Size([32, 1])
    print(f"Prediction value range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
