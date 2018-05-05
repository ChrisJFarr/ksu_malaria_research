from imblearn.over_sampling import SMOTE


class SMOTER:
    def __init__(self, *args, **kwargs):
        self.smote = SMOTE(*args, **kwargs)
        self.params = dict()
        for key, value in kwargs.items():
            self.params[key] = value

    def fit(self, X, y):
        self.smote.fit(X, y)
        return None

    def transform(self, X, y=None):
        return self.smote.sample(X, y)

    def get_params(self, deep):
        return self.params
