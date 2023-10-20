class Oracle:
    def __init__(self, alpha=0):
        self.alpha = alpha

    def __call__(self, preds, targets):
        raise NotImplementedError
