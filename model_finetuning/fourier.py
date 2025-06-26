import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFTWrapper(nn.Module):
    def __init__(self, base_layer: nn.Linear, n: int = 100, alpha: float = 300.0,
                 freq_bias: str = "middle", bandwidth: float = 0.2):
        super().__init__()
        self.base = base_layer
        self.alpha = alpha
        self.d1, self.d2 = base_layer.weight.shape
        self.n = min(n, self.d1 * self.d2)
        # Инициализация спектральной матрицы с bias
        E = self._sample_entries_with_frequency_bias(freq_bias, bandwidth)
        self.register_buffer("E", E)
        # Обучаемые спектральные коэффициенты
        self.delta_W = nn.Parameter(torch.randn(self.n))

        # Заморозка base-слоя
        for param in self.base.parameters():
            param.requires_grad = False

    def _sample_entries_with_frequency_bias(self, freq_bias, bandwidth):
        bias_map = {
            "low": 0.1,
            "middle": 0.5,
            "high": 0.9,
            "none": None  # равномерное распределение
        }
        fc = bias_map.get(freq_bias, 0.5)
        d1, d2 = self.d1, self.d2

        # Координаты в частотной области
        u = torch.arange(d1).unsqueeze(1).expand(d1, d2)
        v = torch.arange(d2).unsqueeze(0).expand(d1, d2)
        if fc is None:
            prob = torch.ones(d1, d2)
        else:
            # Центр в частотной матрице
            center_u = d1 * fc
            center_v = d2 * fc
            D = ((u - center_u) ** 2 + (v - center_v) ** 2).sqrt()
            DW = bandwidth * max(d1, d2)
            prob = torch.exp(-((D - fc * max(d1, d2)) / DW) ** 2)
        # Нормализация
        prob = prob / prob.sum()
        prob_flat = prob.flatten()

        # Сэмплирование n индексов
        indices = torch.multinomial(prob_flat, self.n, replacement=False)
        E0 = indices // d2
        E1 = indices % d2
        return torch.stack([E0, E1], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)

        device = x.device
        delta_W = self.delta_W.to(device)
        E0 = self.E[0].to(device)
        E1 = self.E[1].to(device)

        F = torch.zeros(self.d1, self.d2, dtype=torch.float32, device=device)
        F[E0, E1] = delta_W

        delta_W_matrix = torch.fft.ifft2(F).real * self.alpha

        if x.shape[-1] != delta_W_matrix.shape[0]:
            raise ValueError(f"[FourierFT] Shape mismatch: x={x.shape}, delta={delta_W_matrix.shape}")

        h += torch.einsum("...d,dc->...c", x, delta_W_matrix)
        return h

def inject_fourierft(model, n=1000, alpha=300.0, freq_bias="middle", bandwidth=0.2):
    total = 0
    for i, layer in enumerate(model.decoder.model.decoder.layers):
        for name in ["v_proj", "q_proj"]:
            target = getattr(layer.self_attn, name)
            if isinstance(target, nn.Linear):
                ft = FourierFTWrapper(
                    target,
                    n=n,
                    alpha=alpha,
                    freq_bias=freq_bias,  # "low", "middle", "high", "none"
                    bandwidth=bandwidth
                )
                setattr(layer.self_attn, name, ft)
                total += 1
    print(f"Всего вставлено адаптеров: {total}")

def freeze_all_except_fourier(model):
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if "delta_W" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Заморожено параметров: {frozen:,}, обучаемо (Fourier): {trainable:,}")
