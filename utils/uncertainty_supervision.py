"""Training-only scale weighting for EMCAD deep supervision."""

import torch


class UncertaintyScaleWeighter:
    """Track per-head predictive entropy and return bounded scale weights."""

    def __init__(self, temperature=0.25, ema_decay=0.9):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.temperature = float(temperature)
        self.ema_decay = float(ema_decay)
        self.entropy_ema = None

    @staticmethod
    def _normalized_entropy(logits):
        if logits.shape[1] == 1:
            probability = torch.sigmoid(logits.float()).clamp(1e-7, 1.0 - 1e-7)
            entropy = -(
                probability * probability.log()
                + (1.0 - probability) * (1.0 - probability).log()
            )
            normalizer = 0.6931471805599453
        else:
            probability = torch.softmax(logits.float(), dim=1).clamp_min(1e-7)
            entropy = -(probability * probability.log()).sum(dim=1)
            normalizer = torch.log(
                torch.tensor(float(logits.shape[1]), device=logits.device)
            )
        return (entropy.mean() / normalizer).detach().clamp(0.0, 1.0)

    @torch.no_grad()
    def update(self, outputs):
        if len(outputs) != 4:
            raise ValueError("uncertainty-weighted deep supervision expects four outputs")
        current = torch.stack([self._normalized_entropy(output) for output in outputs])
        if self.entropy_ema is None:
            self.entropy_ema = current
        else:
            self.entropy_ema = (
                self.ema_decay * self.entropy_ema
                + (1.0 - self.ema_decay) * current
            )
        scores = torch.softmax(-self.entropy_ema / self.temperature, dim=0)
        # Scale by the number of heads so the mean weight remains one. The
        # bounded mixture keeps four heads in [0.8, 1.6] and preserves the
        # original total supervision weight (the weighted loss can differ).
        weights = len(outputs) * (0.25 + 0.2 * (scores - 0.25))
        return weights.detach()
