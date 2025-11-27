import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- surrogate spike + LIF ----------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thr):
        out = (x >= thr).float()
        ctx.save_for_backward(x, thr)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, thr = ctx.saved_tensors
        width = 1.0
        grad = (1 - (x - thr).abs() / width).clamp(min=0.0)
        return grad_output * grad, None


def spike_fn(v, thr):
    return SurrogateSpike.apply(v, thr)


class LIFCell(nn.Module):
    """
    Iterative LIF 
    """
    def __init__(self, size, tau_m=2.0, v_th=1.0, v_reset=0.0):
        super().__init__()
        self.size = size
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.register_buffer('v', torch.zeros(1, *size))

    def forward(self, I, reset_state=False):
        """
        I: [B, T, *size]
        return:
            spikes: [B, T, *size]
            vtraj:  [B, T, *size]
        """
        B, T = I.shape[0], I.shape[1]
        if reset_state or self.v.shape[0] != B:
            self.v = torch.zeros(B, *self.size, device=I.device)

        spikes, vtraj = [], []
        alpha = torch.exp(torch.tensor(-1.0 / self.tau_m, device=I.device))
        thr = torch.tensor(self.v_th, device=I.device)

        for t in range(T):
            self.v = alpha * self.v + I[:, t]
            s = spike_fn(self.v, thr)
            self.v = self.v * (1 - s) + self.v_reset * s
            spikes.append(s)
            vtraj.append(self.v)

        return torch.stack(spikes, 1), torch.stack(vtraj, 1)


# ---------- tau-BN ----------
class TauBN1d(nn.Module):
   
    def __init__(self, num_features, tau_m=2.0, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps,
                                 momentum=momentum, affine=True,
                                 track_running_stats=True)
        self.tau_m = tau_m

    def forward(self, x):
        # x: [B, F]
        y = self.bn(x)
        factor = 1.0 / (1.0 - 1.0 / self.tau_m)
        return y * factor


# ---------- Spiking Conv  ----------
class SConv1DLayer(nn.Module):

    def __init__(self, input_len, tau_m=2.0, T=4):
        super().__init__()
        self.input_len = input_len
        self.T = T
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.bn = TauBN1d(num_features=input_len, tau_m=tau_m)
        self.lif = LIFCell(size=(input_len,), tau_m=tau_m)

    def forward(self, x):
        """
        x: [B, input_len]
        return: [B, input_len]
        """
        z = self.conv(x.unsqueeze(1)).squeeze(1)  # [B, L]
        z = self.bn(z)

        I = z.unsqueeze(1).repeat(1, self.T, 1)   # [B, T, L]
        spikes, _ = self.lif(I, reset_state=True)

        return spikes.mean(dim=1)                 # [B, L]


# ---------- Primary Capsule ----------
class SharedPrimaryCaps(nn.Module):
   
    def __init__(self, input_dim, num_capsules=8, caps_dim=8, T=10):
        super().__init__()
        self.T = T
        self.num_caps = num_capsules
        self.caps_dim = caps_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 2 * caps_dim),
            nn.GLU()
        )

        self.heads = nn.Parameter(
            torch.randn(num_capsules, caps_dim, caps_dim) * 0.02
        )

        self.lif = LIFCell(size=(num_capsules, caps_dim))

    def forward(self, x):
        """
        x: [B, input_dim]
        return spikes: [B, T, num_caps, caps_dim]
        """
        base = self.shared(x)  # [B, caps_dim]
        proj = torch.einsum('b d, k d h -> b k h', base, self.heads)  # [B, K, caps_dim]
        I = proj.unsqueeze(1).repeat(1, self.T, 1, 1)  # [B, T, K, caps_dim]
        spikes, _ = self.lif(I, reset_state=True)
        return spikes


# ---------- STDP Digit Capsule ----------
class STDPDigitCaps(nn.Module):

    def __init__(
        self,
        in_caps=8,
        in_dim=8,
        out_caps=2,
        out_dim=16,
        T=10,
        tau_pre=2.0,
        tau_post=2.0,
        eta=0.05,
    ):
        super().__init__()
        self.in_caps, self.in_dim = in_caps, in_dim
        self.out_caps, self.out_dim = out_caps, out_dim
        self.T = T

        # [out_caps, in_caps, out_dim, in_dim]
        self.W = nn.Parameter(
            torch.randn(out_caps, in_caps, out_dim, in_dim) * 0.05
        )
        self.tau_pre, self.tau_post, self.eta = tau_pre, tau_post, eta

    def forward(self, primary_spikes):
        """
        primary_spikes: [B, T, in_caps, in_dim]
        return v: [B, out_caps, out_dim]
        """
        B, T = primary_spikes.shape[0], primary_spikes.shape[1]

        # u_hat: [B, T, out_caps, out_dim]
        u_hat = torch.einsum('o i h d, b t i d -> b t o h', self.W, primary_spikes)

        b = torch.zeros(B, self.in_caps, self.out_caps, device=u_hat.device)
        pre_trace = torch.zeros(B, self.in_caps, device=u_hat.device)
        post_trace = torch.zeros(B, self.out_caps, device=u_hat.device)

        a_pre = torch.exp(torch.tensor(-1.0 / self.tau_pre, device=u_hat.device))
        a_post = torch.exp(torch.tensor(-1.0 / self.tau_post, device=u_hat.device))
        thr_post = 0.5

        for t in range(T):
            pre = primary_spikes[:, t].sum(dim=-1)   # [B, in_caps]
            post_act = u_hat[:, t].norm(dim=-1)      # [B, out_caps]
            post = (post_act >= thr_post).float()    # [B, out_caps]

            pre_trace = a_pre * pre_trace + pre
            post_trace = a_post * post_trace + post

            delta = (
                torch.einsum('b i, b o -> b i o', pre_trace, post)
                + torch.einsum('b o, b i -> b i o', post_trace, pre)
            )
            b = b + self.eta * delta

        c = torch.softmax(b, dim=-1)                 # [B, in_caps, out_caps]
        s = torch.einsum('b i o, b t o h -> b t o h', c, u_hat)  # [B, T, out_caps, out_dim]
        v = s.mean(dim=1)                            # [B, out_caps, out_dim]
        return v


# ---------- Attention Digit Capsule ----------
class AttnDigitCaps(nn.Module):
    def __init__(self, in_caps=8, in_dim=8, out_caps=2, out_dim=16):
        super().__init__()
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.out_caps = out_caps
        self.proj_out = nn.Linear(out_dim, out_dim)

    def forward(self, primary_spikes):
        """
        primary_spikes: [B, T, in_caps, in_dim]
        return v: [B, out_caps, out_dim]
        """
        x = primary_spikes.mean(1)  # [B, in_caps, in_dim]
        Q, K, V = self.query(x), self.key(x), self.value(x)  # [B, I, Do]
        attn = torch.softmax(
            Q @ K.transpose(1, 2) / (K.size(-1) ** 0.5), dim=-1
        )  # [B, I, I]
        U = attn @ V  # [B, I, Do]

        chunks = torch.chunk(U, self.out_caps, dim=1)
        v = torch.stack([c.mean(1) for c in chunks], dim=1)  # [B, out_caps, out_dim]
        return self.proj_out(v)


# ---------- Per-Class Readout ----------
class PerClassReadout(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_classes)])
        self.gate = nn.Parameter(torch.zeros(num_classes, dim))
        self.cls = nn.Linear(num_classes * dim, num_classes)

    def forward(self, v):
        """
        v: [B, C, D]
        """
        outs = []
        for c in range(v.size(1)):
            h = self.ln[c](v[:, c, :])
            g = torch.sigmoid(self.gate[c]).unsqueeze(0) * h
            outs.append(g)
        z = torch.cat(outs, dim=-1)  # [B, C*D]
        logits = self.cls(z)
        return logits


# ---------- Spiking Conv Capsule Head ----------
class SCCapsNetHead(nn.Module):
    """
    structure:
        x 
          ->  SConv1DLayer
          -> SharedPrimaryCaps
          -> STDPDigitCaps + AttnDigitCaps
          -> PerClassReadout / L2-norm-length
    """
    def __init__(
        self,
        input_dim,
        num_classes=2,
        T=10,
        num_primary=8,
        primary_dim=8,
        digit_dim=16,
        use_fc_head=True,
        use_sconv=True,
        sconv_layers=3,
        sconv_T=4,
        lif_tau=2.0,
    ):
        super().__init__()
        self.use_fc_head = use_fc_head
        self.use_sconv = use_sconv

        if use_sconv and sconv_layers > 0:
            sconv_list = []
            for _ in range(sconv_layers):
                sconv_list.append(SConv1DLayer(input_dim, tau_m=lif_tau, T=sconv_T))
            self.sconv = nn.Sequential(*sconv_list)
            caps_input_dim = input_dim
        else:
            self.sconv = None
            caps_input_dim = input_dim

        self.primary = SharedPrimaryCaps(
            caps_input_dim,
            num_capsules=num_primary,
            caps_dim=primary_dim,
            T=T,
        )
        self.stdp_digit = STDPDigitCaps(
            in_caps=num_primary,
            in_dim=primary_dim,
            out_caps=num_classes,
            out_dim=digit_dim,
            T=T,
        )
        self.attn_digit = AttnDigitCaps(
            in_caps=num_primary,
            in_dim=primary_dim,
            out_caps=num_classes,
            out_dim=digit_dim,
        )

        if use_fc_head:
            self.readout = PerClassReadout(num_classes, digit_dim)

    def forward(self, x):
        """
        x: [B, input_dim]
        return:
            probs:  [B, C]
            v:      [B, C, D]
            logits: [B, C]  (若 use_fc_head=False，则为 capsule length)
        """
        if self.sconv is not None:
            x = self.sconv(x)              # [B, input_dim]

        spikes = self.primary(x)           # [B, T, P, Dp]
        v_stdp = self.stdp_digit(spikes)   # [B, C, Dc]
        v_attn = self.attn_digit(spikes)   # [B, C, Dc]
        v = v_stdp + v_attn                # [B, C, Dc]

        if self.use_fc_head:
            logits = self.readout(v)       # [B, C]
            probs = F.softmax(logits, dim=-1)
            return probs, v, logits
        else:
            caps_length = v.norm(dim=-1)   # [B, C]
            probs = F.softmax(caps_length, dim=-1)
            return probs, v, caps_length
