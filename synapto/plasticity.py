class PlasticityController:
    """
    Управляет параметром пластичности P [-1.0 .. 2.0].
    """
    def __init__(self, p_value: float = 1.0, base_lr: float = 2e-5):
        self.base_lr = base_lr
        self.set_plasticity(p_value)

    def set_plasticity(self, p_value: float) -> None:
        self.p_value = max(-1.0, min(2.0, float(p_value)))
        if self.p_value <= -1.0:
            self.lr = 0.0
            self.threshold = float('inf')
            self.is_frozen = True
        else:
            self.lr = self.base_lr * (self.p_value + 1.0)
            self.threshold = 1.0 / (1.0 + max(0.0, self.p_value))
            self.is_frozen = False