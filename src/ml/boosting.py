import xgboost
import lightgbm
from xgboost.callback import EarlyStopping

class BoostingClassifier:
    def __init__(self, model_type='', path_model='', **kwargs):
        """
        A wrapper class for XGBoost and LightGBM classifiers.

        Parameters:
        - model_type (str): "xgboost" or "lightgbm"
        - kwargs: Additional arguments passed to the classifiers.
        """
        self.model_type = model_type
        if model_type == "xgboost":
            if not path_model:
                self.model = xgboost.XGBClassifier(**kwargs)     
            else:
                self.model = xgboost.Booster(model_file=path_model)          
        elif model_type == "lightgbm":
            if not path_model:
                self.model = lightgbm.LGBMClassifier(**kwargs)
            else:
                self.model = lightgbm.Booster(model_file=path_model)
        else:
            raise ValueError("model_type must be 'xgboost' or 'lightgbm'")

    def fit(self, X, y, **kwargs):
        """Fit the model."""
        if self.model_type=='xgboost':
            kwargs.pop('eval_names',None)
            kwargs.pop('eval_metric',None)
            kwargs.pop('callbacks',None)
            kwargs['eval_set'] = kwargs['eval_set'][:1] # only validation set for xgboost
            self.model.fit(X, y, verbose=False, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Make predictions."""
        if self.model_type=='xgboost': 
            X = xgboost.DMatrix(X, feature_names=list(X.columns))
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X, y, sample_weight=None) -> float:
        """Return the mean accuracy on the given test data and labels."""
        return self.model.score(X,y,sample_weight=sample_weight)

    def get_model(self):
        """Return the underlying model instance."""
        return self.model
            
    def save_model(self,path):
        if self.model_type == 'xgboost':
            path = path+'.json'
            self.model.save_model(path)
        elif self.model_type == "lightgbm":
            path = path+'.txt'
            self.model.booster_.save_model(path)  
        return path          
        
    def get_early_stopping(self,stopping_rounds):
        if self.model_type == "xgboost":
            return EarlyStopping(
                                                rounds=stopping_rounds,
                                                min_delta=1e-3,
                                                save_best=True,
                                                maximize=False,
                                                )
        elif self.model_type == "lightgbm":
            return lightgbm.early_stopping(stopping_rounds, verbose=False)