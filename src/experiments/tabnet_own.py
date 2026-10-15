from typing import List
from ml.tabnet import Tabnet
from data.own import Dataset_Own
from optuna import Trial
from pandas import DataFrame
import utils as ut

class Tabnet_Own(Tabnet):
    def __init__(self, dataset:Dataset_Own, options:dict):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial:Trial):
        params_data = {}
        # factor
        params_data = self._set_factor(params_data,trial)
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}

    def set_hyperparams_test(self, params:dict) -> dict:
        """
        There are no hyperparams to set in this test's experiment.
        """
        params_data = {
            "data" : {
                "factor_flag":params.get("factor_flag",False),
                "factor_scaler":params.get("factor_scaler",None),
                "factor":params.get("factor",1),            
            }
        }
        return params_data
    
    def prepare_trial_data(self, dataset: Dataset_Own, params:dict = {}, trial:Trial=False) -> List[DataFrame]:
        self.log_print("INFO: Preparing trial data...")
        _,train,val,test = dataset.split_data()
        # Balance condition
        if dataset.options.get("balance",False): 
            train = dataset._balance_classes(train,'label')
            val = dataset._balance_classes(val,'label')
            test = dataset._balance_classes(test,'label')
        # Factor application
        labels = 'label'
        if params["factor_flag"]:
            labels = ['draw','home','away']
            train.loc[:,labels] = self.factor(train,ut.dif_result(train),params["factor"],params["factor_scaler"])
            val.loc[:,labels]   = self.factor(val,ut.dif_result(val),params["factor"],params["factor_scaler"])
            test.loc[:,labels]  = self.factor(test,ut.dif_result(test),params["factor"],params["factor_scaler"])

        X_train, y_train = dataset.split_data_labels(train,labels)
        X_val, y_val = dataset.split_data_labels(val,labels)
        X_test, y_test = dataset.split_data_labels(test,labels)
        if trial:
            trial.set_user_attr("len_train",len(X_train))
            trial.set_user_attr("len_val",len(X_val))
            trial.set_user_attr("len_test",len(X_test))
            trial.set_user_attr('dims',len(X_train.columns))
        return test,X_train,y_train,X_val,y_val,X_test,y_test