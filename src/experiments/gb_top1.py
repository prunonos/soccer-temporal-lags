
from typing import Dict, List

from optuna import Trial
from pandas import DataFrame
from data.top1 import Dataset_top1
from ml.gradientboosting import GradientBoosting


class GB_top1(GradientBoosting):
    def __init__(self, model_type: str, dataset: Dataset_top1, options: Dict):
        super().__init__(model_type, dataset, options)

    def set_hyperparams(self, trial: Trial):
        params_data = {
            "lamda": trial.suggest_float("lamda", 0.005, 0.1), 
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),       
        }
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}
    
    def set_hyperparams_test(self, params: Dict[str, Dict]):
        params_data = {
            "data" : {
                "lamda": params["lamda"], 
                "gamma": params["gamma_pi"],       
            }
        }
        return params_data       
    
    def prepare_trial_data(self, dataset: Dataset_top1, params: Dict, trial:Trial=False) -> List[DataFrame]:
        dataset.ensamble_data(params["lamda"],params["gamma"])
        _,train,val,test = dataset.split_data()
        # Balance condition
        if dataset.options.get("balance",False): 
            train = dataset._balance_classes(train,'label')
            val = dataset._balance_classes(val,'label')
            test = dataset._balance_classes(test,'label')
        X_train, y_train = dataset.split_data_labels(train)
        X_val, y_val = dataset.split_data_labels(val)
        X_test, y_test = dataset.split_data_labels(test)
        if trial:
            trial.set_user_attr("len_train",len(X_train))
            trial.set_user_attr("len_val",len(X_val))
            trial.set_user_attr("len_test",len(X_test))
        return test,X_train,y_train,X_val,y_val,X_test,y_test