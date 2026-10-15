
import copy
from typing import Dict, List

from optuna import Trial
from pandas import DataFrame
from data.ratings_berrar import Dataset_ratings_berrar
from ml.gradientboosting import GradientBoosting


class GB_ratings_berrar(GradientBoosting):
    def __init__(self, dataset: Dataset_ratings_berrar, options: Dict):
        super().__init__('xgboost', dataset, options)

    def set_hyperparams(self, trial: Trial):
        data_params = self.options['data'].get('ratings',dict()).get('params',{})
        if data_params: 
            print("Data hyperparams provided: ", data_params)
            for k,v in data_params.items(): trial.study.set_user_attr(k,v)
        params_data = {
                'alphaH':   data_params['alphaH'] if 'alphaH' in data_params.keys() else trial.suggest_float("alphaH", 4, 7),
                'alphaA':   data_params['alphaA'] if 'alphaA' in data_params.keys() else trial.suggest_float("alphaA", 4, 7),
                'betaH':    data_params.get('betaH',   trial.suggest_float("betaH", 0., 5.)),
                'betaA':    data_params.get('betaA',   trial.suggest_float("betaA", 0., 5.)),
                'phiH':     data_params.get('phiH',    trial.suggest_float("phiH", -5., 5.)),
                'phiA':     data_params.get('phiA',    trial.suggest_float("phiA", -5., 5.)),
                'weightHomeAttack': data_params.get('weightHomeAttack', trial.suggest_float("weightHomeAttack", 0, 1.5)),
                'weightHomeDef':    data_params.get('weightHomeDef',    trial.suggest_float("weightHomeDef", 0, 1.5)),
                'weightAwayAttack': data_params.get('weightAwayAttack', trial.suggest_float("weightAwayAttack", 0, 1.5)),
                'weightAwayDef':    data_params.get('weightAwayDef',    trial.suggest_float("weightAwayDef", 0, 1.5)),
        }
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}
    
    def set_hyperparams_test(self, params: Dict[str, Dict]):
        data_params = self.options['data'].get('ratings',dict()).get('params',{})
        params_data = {
            "data" : {
                'alphaH': params.get('alphaH',data_params['alphaH']),
                'alphaA': params.get('alphaA',data_params['alphaA']),
                'betaH': params['betaH'],
                'betaA': params['betaA'],
                'phiH': params['phiH'],
                'phiA': params['phiA'],
                'weightHomeAttack': params['weightHomeAttack'],
                'weightHomeDef': params['weightHomeDef'],
                'weightAwayAttack': params['weightAwayAttack'],
                'weightAwayDef': params['weightAwayDef'],
            }
        }
        return params_data       
    
    def prepare_trial_data(self, dataset: Dataset_ratings_berrar, params: Dict, trial:Trial=False) -> List[DataFrame]:
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.train_pi_rates(dataset_copy.data,**params)
        _,train,val,test = dataset_copy.split_data()
        X_train, y_train = dataset_copy.split_data_labels(train)
        X_val, y_val = dataset_copy.split_data_labels(val)
        X_test, y_test = dataset_copy.split_data_labels(test)
        if trial:
            trial.set_user_attr("len_train",len(X_train))
            trial.set_user_attr("len_val",len(X_val))
            trial.set_user_attr("len_test",len(X_test))
        return test,X_train,y_train,X_val,y_val,X_test,y_test