from typing import Any, Dict
from optuna import Trial
from data.owntop1 import Dataset_OwnTop1
from ml.tabnet import Tabnet
import utils as ut

class Tabnet_Owntop1(Tabnet):
    def __init__(self, dataset, options):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial:Trial) -> Dict[str,Any]:
        params_data = {
            "lamda": trial.suggest_float("lamda", 0.005, 0.1), 
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),       
        }
        # factor
        params_data = self._set_factor(params_data,trial)
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}
    
    def set_hyperparams_test(self, params:Dict[str,Any]) -> Dict[str,Any]:
        params_data = {
            "data" : {
                "lamda": params["lamda"], 
                "gamma": params["gamma_pi"], 
                "factor_flag":params.get("factor_flag",False),
                "factor_scaler":params.get("factor_scaler",None),
                "factor":params.get("factor",1),            
            }
        }
        return params_data       

    def prepare_trial_data(self, dataset: Dataset_OwnTop1, params: Dict, trial: Trial = False):
        self.log_print("Preparing trial data...")
        dataset.ensamble_data(params["lamda"],params["gamma"])
        _,train,val,test = dataset.split_data()
        # Balance condition
        if dataset.options.get("balance",False): 
            train = dataset._balance_classes(train,'label')
            val = dataset._balance_classes(val,'label')
            test = dataset._balance_classes(test,'label')
        # Factor application
        if params["factor_flag"]:
            train.loc[:,'label'] = ut.update_labels(
                self.factor(train,ut.dif_result(train),params["factor"],params["factor_scaler"])
            )
            val.loc[:,'label'] = ut.update_labels(
                self.factor(val,ut.dif_result(val),params["factor"],params["factor_scaler"])
            )
            test.loc[:,'label'] = ut.update_labels(
                self.factor(test,ut.dif_result(test),params["factor"],params["factor_scaler"])
            )
        X_train, y_train = dataset.split_data_labels(train)
        X_val, y_val = dataset.split_data_labels(val)
        X_test, y_test = dataset.split_data_labels(test)
        if trial:
            trial.set_user_attr("len_train",len(X_train))
            trial.set_user_attr("len_val",len(X_val))
            trial.set_user_attr("len_test",len(X_test))
            trial.set_user_attr('dims',len(X_train.columns))
        return test,X_train,y_train,X_val,y_val,X_test,y_test