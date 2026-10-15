import copy

from optuna import Trial
from data.pi_ratings import Dataset_piRatings
from ml.tabnet import Tabnet

COLS_META   = ["matchId","Date","Div","HomeTeam","AwayTeam","label"]

class Tabnet_piRatings(Tabnet):
    def __init__(self, dataset, options):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial):
        self.log_print(f"Setting trial nº {trial.number} parameters...")
        params_data = {
            "lamda": trial.suggest_float("lamda", 0.005, 0.1), 
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),       
        }
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}

    def set_hyperparams_test(self, params):
        params_data = {
            "data" : {
                "lamda": params["lamda"], 
                "gamma": params["gamma_pi"],       
            }
        }
        return params_data       
    
    def prepare_trial_data(self,dataset:Dataset_piRatings,params,trial:Trial=False):
        self.log_print("Preparing trial data...")
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.train_pi_rates(dataset_copy.data,params["lamda"],params["gamma"])
        # dataset_copy.data = dataset_copy.data[dataset_copy.features]
        _,train,val,test = dataset_copy.split_data()
        # return input_data (X_train,y_train,X_val,y_val,X_test,y_test)
        X_train, y_train = dataset_copy.split_data_labels(train)
        X_val, y_val = dataset_copy.split_data_labels(val)
        X_test, y_test = dataset_copy.split_data_labels(test)
        if trial:
            trial.set_user_attr("len_train",len(X_train))
            trial.set_user_attr("len_val",len(X_val))
            trial.set_user_attr("len_test",len(X_test))
        return test,X_train,y_train,X_val,y_val,X_test,y_test
