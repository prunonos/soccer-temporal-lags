import os
from typing import List, Tuple
from optuna import Study, Trial
import pandas as pd
import torch
import utils as ut
from data.base import Dataset
from ml.experiment import Experiment
from pytorch_tabnet.tab_model import TabNetClassifier, TabModel
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.metrics import Metric
from torch.utils.tensorboard import SummaryWriter

RPS_METRIC = lambda metric: RPS if 'rps' in metric else metric

class Tabnet(Experiment):
    def __init__(self,dataset,options):
        super().__init__(dataset,options)
        self.unsupervised_training = self.train_config["unsupervised_training"] # boolean
        self.feature_importance = []
        self.process_data(dataset) # no devuelve nada
        tune = self.tuning(dataset)
        accuracy = self.test(tune,dataset)
        self.save_metrics(tune,accuracy)
        self.config_experiment(tune,dataset)

    def prepare_model(self, params, data=None):
        model = TabNetClassifier(**params["tabnet"])
        unsupervised_model = None
        if self.unsupervised_training:
            unsupervised_model = TabNetPretrainer(**params["unsupervised"])
        return model, unsupervised_model

    def fit(self, models:List[TabModel], data:List[pd.DataFrame], trial: Trial, param_grid:dict) -> Tuple[TabNetClassifier,float]:
        X_train, y_train, X_valid, y_valid = map(lambda x: x.values,data[1:5])
        clf, unsupervised_model = models

        if self.unsupervised_training and unsupervised_model!=None:
            self.log_print("Unsupervised training running...")
            unsupervised_model.fit(
                X_train=X_train,
                eval_set=[X_valid],
                max_epochs=int(self.train_config["max_epochs"]/2),
                pretraining_ratio=param_grid["pretraining_ratio"],
                batch_size=param_grid["batch_size"], virtual_batch_size=param_grid["virtual_batch_size"]/2,
                drop_last=False
            )

        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=list(map(RPS_METRIC,self.train_config["eval_metric"])), # si es 'rps' debemos pasarle la clase RPS definida
            max_epochs=self.train_config["max_epochs"],
            patience=self.train_config["patience"],
            batch_size=param_grid["batch_size"], virtual_batch_size=param_grid["virtual_batch_size"]/2,
            drop_last=False,
            from_unsupervised=unsupervised_model, # None if non-pretraining
        ) 

        accuracy, _ = clf.best_cost, clf.best_epoch
        # Loguear métricas en TensorBoard
        self.writer = SummaryWriter(log_dir=f"{os.getcwd()}{self.options['paths']['logger']}{self.exp_id}_{self.now}/version_{self.version}")
        metric_dict = self.log_metrics_to_tensorboard(clf,self.train_config["eval_metric"])
        self.writer.add_hparams(hparam_dict=trial.params,metric_dict=metric_dict)
        # get historical metrics
        self.save_best_epoch_metrics_to_trial(clf,trial,self.train_config["eval_metric"])
        # get feat importance    
        self.feature_importance = pd.DataFrame({'features':data[1].columns,'importance':clf.feature_importances_}).sort_values("importance",ascending=False) # data[1] == train data
        return clf,accuracy
    
    def test(self,study:Study,dataset:Dataset) -> float:
        model = self.load_best_model(study)
        param_grid = self.set_hyperparams_test(study.best_params)
        data_input = self.prepare_trial_data(dataset, param_grid["data"])
        metric,logits = self.predict(model,data_input)
        self.save_logits(data_input[0],logits) # data_input[0] == test completo con labels
        if self.options['data'].get('save_data',False): self.save_data('test',data_input[0],study) # data_input[0] == test completo con labels
        return metric   

    def predict(self, clf:TabNetClassifier, data:List[pd.DataFrame]) -> Tuple[float,pd.DataFrame]:
        """
        Input:
            - models: we only take the supervised trained classifier.
            - data: receive all dataframes but we only take the test set.
        Returns:
            - accuracy: test accuracy
            - logits: dataframe with matchId and logits
        """
        X_test, y_test = data[-2:]
        logits = clf.predict_proba(X_test.values)
        preds = logits.argmax(axis=1)
        accuracy = (preds==y_test.values).mean()
        logits = pd.DataFrame({"draw":logits[:,0],"home":logits[:,1],"away":logits[:,2]},index=X_test.index)
        return accuracy, logits 
    
    def set_hyperparams(self,trial):
        tabnet_grid = {   
            "tabnet": {
                "n_d": trial.suggest_int("n_d",8,64,log=True),
                "n_a": trial.suggest_int("n_a",8,64,log=True),
                "n_steps": trial.suggest_int("n_steps",3,10),
                "gamma": trial.suggest_float("gamma_tabnet",1.0,2.0,log=True),
                "n_independent": trial.suggest_int("n_independent",1,5),
                "n_shared": trial.suggest_int("n_shared",1,5),
                "lambda_sparse":1e-3,
                "optimizer_fn":torch.optim.Adam,
                "optimizer_params": {
                    "lr":trial.suggest_float("lr",1e-3,1,log=True)
                    },
                "scheduler_params": {
                    "gamma": trial.suggest_float("gamma_scheduler",1e-1,1),
                    "step_size": 20
                    },
                "scheduler_fn": torch.optim.lr_scheduler.StepLR
            },
            "batch_size": self.train_config.get("select",trial.suggest_categorical("batch_size",[512,1024,2048,4096,8192])),
            "virtual_batch_size": self.train_config.get("select",128*2)
        }        
        
        if self.unsupervised_training:
            unsup_grid = {
                "unsupervised": {
                    "n_d": trial.params["n_d"],
                    "n_a": trial.params["n_a"],
                    "n_steps": trial.params["n_steps"],
                    "gamma": trial.params["gamma_tabnet"],
                    "n_independent": trial.params["n_independent"],
                    "cat_emb_dim":1,
                    "n_shared": trial.params["n_shared"],
                    "optimizer_fn":torch.optim.Adam,
                    "optimizer_params": {
                        "lr":trial.params["lr"]
                    },
                    "mask_type":trial.suggest_categorical("mask_type",["sparsemax","entmax"])
                },
                "pretraining_ratio": trial.suggest_float("pretraining_ratio",0.2,0.9,log=True),
                "batch_size": self.train_config.get("select",trial.suggest_categorical("batch_size",[512,1024,2048,4096,8192])),
                "virtual_batch_size": self.train_config.get("select",128*2)
            }
            return {**tabnet_grid,**unsup_grid}
        else: 
            return tabnet_grid
        
    def set_best_model_path(self,trial:Trial,clf:TabNetClassifier,metric:float):
        """
        Evaluate if this model is the best one, in that case we save it.
        returns: the best model path
        """
        study = trial.study
        name = f"{self.exp_id}_{self.now}"
        compare = self.get_func_best_model(study)
        if trial.number==0 or compare(clf.best_cost,study.best_value):
            self.log_print("Saving best model...")
            path = clf.save_model(f"./logs/models/{name}/{name}_v{self.version}")
            self.save_feature_importance(self.feature_importance,f"models/{name}/feature_importance_{name}")
            study.set_user_attr("best_model_path",path)
            # study.set_user_attr("model_folder",f"./logs/models/{name}/")
            if trial.number>0: os.remove(f"./logs/models/{name}/{name}_v{study.best_trial.number}.zip")

    def load_best_model(self,study:Study) -> TabNetClassifier:
        best_model_path = study.user_attrs.get("best_model_path")
        model = TabNetClassifier()
        model.load_model(best_model_path)
        return model
    
    def save_feature_importance(self, feature_importance:pd.DataFrame, path:str):
        self.save_dataframe(feature_importance,path)

    def save_logits(self,data:pd.DataFrame,logits:pd.DataFrame):
        """
        Stores the logits (and predictions) in a dataframe with some (extra)metadata about the matches.

        Input:
            - data: dataframe with the whole data
            - logits: dataframe with the logits predicted
        """
        df = logits.join(data,how="inner")
        output = self.format_df_logits(df)
        self.save_dataframe(output,self.exp_id+'_logits')

    def save_best_epoch_metrics_to_trial(self, clf: TabNetClassifier, trial: Trial, eval_metrics: List[str]):
        """
        Extrae las métricas de la mejor época y las guarda en los atributos del usuario de Optuna Trial.
        
        Args:
            clf: Modelo TabNetClassifier entrenado.
            trial: Objeto de Optuna Trial.
        """
        # Obtener la mejor época y el historial
        best_epoch = clf.best_epoch
        history = clf.history
        # Extraer métricas relevantes (entrenamiento y validación)
        for metric in eval_metrics:
            try:
                trial.set_user_attr(f"train_{metric}", round(history[f'train_{metric}'][best_epoch],4))
                trial.set_user_attr(f"val_{metric}", round(history[f'valid_{metric}'][best_epoch],4))
            except Exception:
                self.log_print(f'¡¡¡{metric} does not exist in clf.history!!!')
        # Save loss
        try:
            trial.set_user_attr(f"loss", round(history[f'loss'][best_epoch],4))
        except Exception:
            self.log_print(f'¡¡¡Loss does not exist in clf.history!!!')        

    def log_metrics_to_tensorboard(self, clf: TabNetClassifier, eval_metrics: List[str]):
        """
        Loguea las métricas de todas las épocas en TensorBoard en un único informe.
        """
        best_epoch = clf.best_epoch
        history = clf.history
        epochs = range(1, len(history['loss']) + 1)  # Eje X basado en el número de épocas
        metric_dict = {}
        
        # log metrics
        for metric in eval_metrics:
            if metric != "loss":
                # creamos el dict de metricas - estandarizamos el nombre de las metricas
                key = 'loss' if metric=='logloss' else metric
                metric_dict[f'train_{key}_m'] = history[f"train_{metric}"][best_epoch]
                metric_dict[f'val_{key}_m'] = history[f"valid_{metric}"][best_epoch]
                # Loguear métricas de entrenamiento y validación
                for epoch, value in zip(epochs, history[f"train_{metric}"]):
                    self.writer.add_scalar(f"train_{key}", value, epoch)
                for epoch, value in zip(epochs, history[f"valid_{metric}"]):
                    self.writer.add_scalar(f"val_{key}", value, epoch)
            elif metric == 'loss':
                metric_dict['loss_m'] = history["loss"][best_epoch]
                # Loguear la pérdida general
                for epoch, value in zip(epochs, history[metric]):
                    self.writer.add_scalar("loss", value, epoch)
        # seteamos metricas
        return metric_dict

    def __del__(self):
        """
        Asegúrate de cerrar el logger de TensorBoard al finalizar.
        """
        self.writer.close()

class RPS(Metric):
    def __init__(self):
        self._name = "rps"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return ut.avg_rps(y_true,y_score)
