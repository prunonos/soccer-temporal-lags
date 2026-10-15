import os
from typing import List, Tuple
import numpy as np
from pandas import DataFrame
from ml.experiment import Experiment	
from optuna import Study, Trial, trial
from ml.mlp import MLP
from data.base import Dataset, TorchData

class Own(Experiment):
    def __init__(self, dataset:Dataset, options):
        super().__init__(dataset,options)   # Call Experiment's __init__ method
        self.process_data(dataset) # no devuelve nada
        tune = self.tuning(dataset)
        accuracy = self.test(tune,dataset)
        self.save_metrics(tune,accuracy)
        self.config_experiment(tune,dataset)
        
    def prepare_model(self, params:dict, data=None):
        params["eval_metric"] = self.train_config["eval_metric"]
        params["direction"] = self.train_config["direction"]
        params["patience"] = self.train_config["patience"]
        return MLP(self.exp_id,self.now,self.version,**params)

    def fit(self, model:MLP, data:List[DataFrame], trial:Trial, param_grid=None) -> Tuple[MLP,float]:
        model.set_training_data(*data)
        trainer = model.trainer
        trainer.fit(model)
        metric = model.get_best_score()
        return model,metric

    def test(self, study:Study, dataset):
        model = self.load_best_model(study)
        params = self.set_hyperparams_test(study.best_params)
        data_input = self.prepare_trial_data(dataset,params["data"])
        metric,logits = self.predict(model,data_input)
        # save logits and test data
        _,_,data_test = data_input
        self.save_logits(data_test,logits)
        # get test df from TorchData class
        if self.options['data'].get('save_data',False): 
            df_test = data_test.as_df(labels_encoded=False,data=True,metadata=True)
            self.save_data('test',df_test,study) # data_input[0] == test completo con labels
        return metric
    
    def predict(self, model:MLP, data):
        model.set_training_data(*data)
        trainer = model.create_trainer(inference_mode=True)
        logits = trainer.predict(model)
        probs = model.get_probs_from_logits(logits)
        metric = model.compute_metrics(probs)
        return metric,probs

    def set_best_model_path(self,trial:Trial,model:MLP,metric:float):
        study = trial.study
        compare = self.get_func_best_model(study)
        if trial.number==0 or compare(model.get_best_score(),study.best_value):
            if trial.number>0: os.remove(study.user_attrs.get("best_model_path",None))
            path = model.get_best_model_path()
            study.set_user_attr("best_model_path",path)
        else:
            os.remove(model.get_best_model_path())

    def load_best_model(self,study:Study):
        best_model_path = study.user_attrs.get("best_model_path")
        return MLP.load_from_checkpoint(best_model_path)
        
    def set_hyperparams(self,trial:trial):
        model_grid = {
            "learning_rate": trial.suggest_float("learning_rate",1e-3,1,log=True),
            "batch_size": trial.suggest_categorical("batch_size", 
                                    self.train_config.get("batch_size",[16,32,64,128])),
            "max_epochs": self.train_config.get("max_epochs",100),
            "num_workers": 2,
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 5),
            "activation": trial.suggest_categorical("activation",
                                                ["relu", "selu", "leaky_relu"]),
            "mode":trial.suggest_categorical("mode", ["vanilla", "batchnorm", "dropout"]), 
        }
        
        n_units_lx = { f"n_units_l{l+1}":trial.suggest_int(f"n_units_l{l+1}", 3, 30) 
                                for l in range(model_grid["num_hidden_layers"]) }
        model_grid["dropout"] = trial.suggest_float("dropout", 0.1, 0.7) if model_grid["mode"]=='dropout' else None
        model_grid["negative_slope"] = trial.suggest_float("neg_slope", 1e-3, 1e-1) if model_grid["activation"]=='leaky_relu' else None
        self.__set_optimizer(model_grid,trial)
        model_grid.update(n_units_lx)
        return model_grid
    
    def __set_optimizer(self,params:dict,trial:Trial):
        optim = trial.suggest_categorical("optim", ["adam", "sgd"])
        if optim=='adam':
            optim_grid = {
                "optim":optim,
                "b1": trial.suggest_float("b1",1e-2,1),
                "b2": trial.suggest_float("b2",1e-2,1)
            }
        elif optim=='sgd':
            optim_grid = {
                "optim":optim,
                "momentum": trial.suggest_float("momentum",1e-3,1,log=True),
                "nesterov": trial.suggest_categorical("nesterov",[True,False])
            }
        params.update(optim_grid)
    
    def save_logits(self, testset:TorchData, logits:np.ndarray):
        test_df = testset.as_df(labels_encoded=False,data=False,metadata=True)
        test_df.loc[:,"draw"] = logits[:,0]
        test_df.loc[:,"home"] = logits[:,1]
        test_df.loc[:,"away"] = logits[:,2]
        output = self.format_df_logits(test_df)
        self.save_dataframe(output,self.exp_id+'_logits')