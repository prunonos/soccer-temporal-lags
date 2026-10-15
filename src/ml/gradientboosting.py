import os
import numpy as np
from optuna import Study, Trial
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from data.base import Dataset
from ml.experiment import Experiment	
from typing import Dict, List, Tuple
from utils import avg_rps
from ml.boosting import BoostingClassifier


class GradientBoosting(Experiment):
    def __init__(self, model_type: str, dataset: Dataset, options: Dict):
        options['training']['model_type'] = model_type
        super().__init__(dataset, options)
        self.model_type = model_type
        self.process_data(dataset)
        tune = self.tuning(dataset)
        tune.set_user_attr('model_type',self.model_type)
        accuracy = self.test(tune,dataset)
        self.save_metrics(tune,accuracy)
        self.config_experiment(tune,dataset)

    def prepare_model(self, params: Dict, data=None):
        return BoostingClassifier(self.model_type,
                                    objective="multiclass",
                                    num_class=3,
                                    **params
                                )

    def fit(self, model:BoostingClassifier, data:List[DataFrame], trial:Trial, param_grid=None) -> float:
        X_train,y_train,X_val,y_val = data[1:5]
        if self.train_config.get("cross_validation",True):
            """ Train with Cross-Entropy """
            data   = pd.concat([X_train,X_val])
            labels = pd.concat([y_train,y_val])
            model,metric,acc = self.cross_validation(model,data,labels,param_grid)
        else:
            """ Unitary training """            
            model,metric,acc = self.train_lgbm(model,X_train,y_train,X_val,y_val,param_grid)
        trial.set_user_attr('acc_train', acc[0])
        trial.set_user_attr('acc_val', acc[1])
        return model,metric
        
    def cross_validation(self,model:BoostingClassifier,X:DataFrame,y:DataFrame,params:Dict):
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        cv_models = []
        cv_scores = []
        for _, (train_idx, test_idx) in enumerate(cv.split(X, y.astype(int))):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model,score,acc = self.train_lgbm(model,X_train,y_train,X_test,y_test,params)
            cv_models.append(model)
            cv_scores.append(score)
        
        best_split = np.array(cv_scores).argmax()
        best_model = cv_models[best_split]
        return best_model,np.mean(cv_scores),acc
    
    def eval_metric(self,pred,truth) -> str:
        metric = self.train_config.get("eval_metric",'multi_logloss')
        if 'rps' in metric: return avg_rps(pred,truth)
        else: return log_loss(truth,pred) # same as cross-entropy

    def train_lgbm(self,model:BoostingClassifier,X_train:DataFrame, y_train:DataFrame, X_val:DataFrame,y_val:DataFrame,params:Dict) -> Tuple[BoostingClassifier,float]:
        # Callbacks
        callbacks = [model.get_early_stopping(
                                    stopping_rounds=params.get("stopping_rounds",10),
                                )]
        if self.model_type=='xgboost': params['callbacks'] = callbacks
        model = self.prepare_model(params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val,y_val),(X_train,y_train)],
            eval_names=['validation','training'],
            eval_metric="multi_logloss",
            callbacks=callbacks
        )
        # TODO: ¿no incluye model ya un atributo del 'eval_score' o hay que añadir a eval_metric la accuracy?
        # score = model.best_score_ # multi_logloss
        preds = model.predict_proba(X_val)
        acc_val = (preds.argmax(axis=1)==y_val.astype(int)).mean()
        acc_train = model.score(X_train,y_train)
        self.log_print(f'Acc. training: {acc_train}')
        self.log_print(f'Acc. validation: {acc_val}')
        score = self.eval_metric(preds,y_val.astype(int)) 
        return model,score, (acc_train,acc_val)

    def test(self,study:Study, dataset) -> float:
        model = self.load_best_model(study)
        param_grid = self.set_hyperparams_test(study.best_params)
        data_input = self.prepare_trial_data(dataset, param_grid["data"])
        metric,logits = self.predict(model,data_input)
        self.save_logits(data_input[0],logits)
        if self.options['data'].get('save_data',False): self.save_data('test',data_input[0],study) # data_input[0] == test completo con labels
        return metric   

    def predict(self, model:BoostingClassifier, data:List[DataFrame]) -> Tuple[float, DataFrame]:
        X_test, y_test = data[-2:]
        logits = model.predict(X_test)
        preds = logits.argmax(axis=1)
        accuracy = (preds==y_test.values).mean()
        logits = pd.DataFrame({"draw":logits[:,0],"home":logits[:,1],"away":logits[:,2]},index=X_test.index)
        return accuracy,logits
    
    def set_hyperparams(self, trial: Trial):
        model_grid = {
            # "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "n_estimators": trial.suggest_categorical("n_estimators", [20,50,100,250,500]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 5, 985, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1050, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1., step=0.1),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [0,1,10,50]),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1., step=0.1),
            "stopping_rounds":trial.suggest_categorical("stopping_rounds",[10,20,50])
        }
        return model_grid

    def set_best_model_path(self,trial:Trial,model:BoostingClassifier,metric:float):
        study = trial.study
        name = f"{self.exp_id}_{self.now}"
        compare = self.get_func_best_model(study)
        if trial.number==0 or compare(metric,study.best_value):
            if trial.number>0: os.remove(study.user_attrs.get("best_model_path",None))
            dir = f"{os.getcwd()}/logs/models/{name}/"
            if not os.path.exists(dir): os.mkdir(dir)
            path = f"{dir}{name}_v{self.version}"
            path = model.save_model(path)
            study.set_user_attr("best_model_path",path)

    def load_best_model(self, study: Study) -> BoostingClassifier:
        best_model_path = study.user_attrs.get("best_model_path")
        model = BoostingClassifier(self.model_type, best_model_path)
        return model

    def save_logits(self,data:pd.DataFrame,logits:pd.DataFrame):
        df = logits.join(data,how="inner")
        # TODO: añadir RPS
        output = self.format_df_logits(df)
        self.save_dataframe(output,self.exp_id+'_logits')