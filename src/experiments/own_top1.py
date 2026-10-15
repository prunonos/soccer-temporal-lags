from optuna import Trial
from data.base import TorchData
from data.owntop1 import Dataset_OwnTop1
from ml.mlp_experiment import Own
import utils as ut


class Own_top1(Own):
    def __init__(self, dataset: Dataset_OwnTop1, options):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial: Trial):
        params_model = super().set_hyperparams(trial)
        params_data = {
            "scaler":trial.suggest_categorical("scaler", ["normalizer","maxabs","minmax"]),
            "lamda": trial.suggest_float("lamda", 0.005, 0.1), 
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),       
        }
        # Dims and feature selection
        feats = self.options["data"].get("num_feats",[])
        method = trial.suggest_categorical("method",["","anova","pca"]) if len(feats) else ""
        if method:
            dims = trial.suggest_int("dims",
                                     min(self.options["dims"],feats[0]),
                                     min(self.options["dims"],feats[1])
                                    )
        else:
            dims = self.options["dims"]
        trial.set_user_attr('dims',dims)
        # factor
        params_data = self._set_factor(params_data,trial)

        params_data.update({"method":method, "dims":dims})
        params_model["dims"] = dims # necesitamos las dims en el build del model
        # build params
        params = {"data":params_data, "model":params_model}
        return params
        
    def set_hyperparams_test(self, params:dict):
        params_data = {
            "scaler":params["scaler"],
            "lamda": params["lamda"], 
            "gamma": params["gamma_pi"],       
            "method":params.get("method",None),
            "dims":params.get("dims",self.options["dims"]),
            "factor_flag":params.get("factor_flag",False),
            "factor_scaler":params.get("factor_scaler",None),
            "factor":params.get("factor",1),            
        }
        return {"data":params_data}
    
    def prepare_trial_data(self, dataset:Dataset_OwnTop1, params:dict, trial:Trial=False):
        self.log_print("Preparing trial data...")
        dataset._set_features()
        dataset.ensamble_data(params["lamda"],params["gamma"])
        _,train,val,test = dataset.split_data()

        if self.train_config.get("select",len(train))<len(train):
            train = train.sample(n=self.options["data"]["select"],random_state=1)
        if self.train_config.get("select",len(val))<len(val):
            val = val.sample(n=self.options["data"]["select"],random_state=1)
        if self.train_config.get("select",len(test))<len(test):
            test = test.sample(n=self.options["data"]["select"],random_state=1)

        train,val,test = dataset.apply_feature_transformation(params["dims"],
                                                              train,val,test,
                                                              method=params["method"]
                                                              )
        
        if dataset.options.get("balance",False): 
            ftr_col = 'FTR' if 'FTR' in dataset.data.columns else 'FTR_H'
            train = dataset._balance_classes(train,ftr_col)
            val = dataset._balance_classes(val,ftr_col)
            test = dataset._balance_classes(test,ftr_col)
        
        # FEATURES CON PCA
        train = dataset.scale_data(train,params['scaler'])
        val   = dataset.scale_data(val,params['scaler'])
        test  = dataset.scale_data(test,params['scaler'])

        # crear un objeto torch.Dataset para cada set
        train = TorchData(train,dataset.features)
        val   = TorchData(val,dataset.features)
        test  = TorchData(test,dataset.features,predict_flag=True)
        if params["factor_flag"]:
            train_factor = train.as_df(data=False,metadata=True)
            train.update_labels(
                self.factor(train_factor,ut.dif_result(train_factor),params["factor"],params["factor_scaler"])
            )
            val_factor = val.as_df(data=False,metadata=True)
            val.update_labels(
                self.factor(val_factor,ut.dif_result(val_factor),params["factor"],params["factor_scaler"])
            )
            test_factor = test.as_df(data=False,metadata=True)
            test.update_labels(
                self.factor(test_factor,ut.dif_result(test_factor),params["factor"],params["factor_scaler"])
            )
        if trial:
            trial.set_user_attr("len_train",len(train))
            trial.set_user_attr("len_val",len(val))
            trial.set_user_attr("len_test",len(test))

        return train, val, test