class Search:
    """
    General interface to be implemented by model structure search methods.
    """
    def predict(self, X):
        return self.mixture_.predict(X)

    def predict_mean_var(self, X):
        return self.mixture_.predict_mean_var(X)

    def predicts(self, X):
        return self.mixture_.predicts(X)
