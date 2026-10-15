from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import utils as ut
from data.base import TorchData

class MLP(pl.LightningModule):
    def __init__(self,exp_id,now,version,**kwargs) -> None:
        super(MLP, self).__init__()
        self.save_hyperparameters()
        # initialize accumulators
        self.metrics = defaultdict(list)
        self.__build()
        self.__get_trainer()
        
    def __build(self):
        # Initialize modules
        self.layers = nn.ModuleList()
        if self.hparams["mode"]=="batchnorm":
            self.batch_norm = nn.ModuleList()
        # INPUT LAYER
        self.layers.append( nn.Linear(self.hparams["dims"],self.hparams["n_units_l1"]) )
        # HIDDEN LAYERS
        for i in range(self.hparams["num_hidden_layers"]-1):
            units = self.hparams[f"n_units_l{i+2}"]
            self.layers.append( nn.Linear(self.layers[-1].out_features,units) )
            if self.hparams["mode"]=="batchnorm":
                # cogemos la (ahora) penultima capa
                self.batch_norm.append( torch.nn.BatchNorm1d(num_features=self.layers[-2].out_features) )
        # OUTPUT LAYER
        if self.hparams["mode"]=="batchnorm":
            self.batch_norm.append( torch.nn.BatchNorm1d(num_features=self.layers[-1].out_features) )
        self.layers.append( nn.Linear(self.layers[-1].out_features,3) )
        # DROP OUT
        if self.hparams["mode"]=="dropout":
            self.dropout = torch.nn.Dropout(p=self.hparams["dropout"])
        # ACTIVATION FUNCTION
        self.activation = self.__get_activation()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.,1.,1.]))

    def __get_trainer(self):
        name = f"{self.hparams.exp_id}_{self.hparams.now}"
        # Configura el logger de TensorBoard
        tensorboard_logger = TensorBoardLogger(
            save_dir='logs/metadata/',
            name=name,
            version=self.hparams.version
        )
        tensorboard_logger.log_hyperparams(
            self.hparams,
            metrics = {
                'train_loss_m': 0,
                'train_accuracy_m': 0,
                'val_loss_m': 0,
                'val_accuracy_m': 0,
                'val_rps_m': 0,
            }
        )        
        self.checkpoint_callback = ModelCheckpoint(
            monitor=self.hparams["eval_metric"],
            mode="max" if self.hparams["direction"]=='maximize' else "min",
            save_top_k=1,
            save_weights_only=False,
            dirpath=f'logs/models/{name}',
            filename=f"{name}_v{self.hparams.version}",
            auto_insert_metric_name=True,
            verbose=True
        )
        early_stopping_callback = EarlyStopping(
            monitor=self.hparams["eval_metric"],
            mode="max" if self.hparams["direction"]=='maximize' else "min",
            min_delta=0.01,
            patience=self.hparams["patience"],
            verbose=True
        )
        self.trainer = self.create_trainer(
            logger=tensorboard_logger,
            max_epochs=self.hparams["max_epochs"],
            callbacks=[self.checkpoint_callback,early_stopping_callback],
            log_every_n_steps=1
        )
        return self.trainer
    
    def create_trainer(self,*args,**kwargs):
        return pl.Trainer(*args,**kwargs)
    
    def get_best_score(self) -> float:
        best_score = self.checkpoint_callback.best_model_score.cpu().item()
        return best_score
    
    # def get_acc_metrics(self) -> Tuple[float, float]:
    #     """
    #     Devuelve las métricas de accuracy (entrenamiento y validación) del mejor checkpoint.
    #     """
    #     # Cargar el mejor modelo desde el checkpoint
    #     best_model_path = self.checkpoint_callback.best_model_path
    #     checkpoint = torch.load(best_model_path)
    #     # Obtener métricas del checkpoint
    #     train_acc = checkpoint.get('callbacks', {}).get('ModelCheckpoint', {}).get('train_accuracy', None)
    #     val_acc = checkpoint.get('callbacks', {}).get('ModelCheckpoint', {}).get('val_accuracy', None)

    #     return train_acc, val_acc
    
    def get_best_model_path(self) -> str:
        return self.checkpoint_callback.best_model_path
    
    def load_best_model(self):
        best_model_path = self.checkpoint_callback.best_model_path
        return self.load_from_checkpoint(best_model_path)
    
    def forward(self, x):
        # input & hidden layers
        for i,layer in enumerate(self.layers[:-1]):
            if self.hparams["mode"]=="dropout":
                x = layer(self.dropout(x))
            elif self.hparams["mode"]=="batchnorm":
                x = self.batch_norm[i](layer(x))
            else:
                x = layer(x)
            x = self.activation(x)
        # output
        if self.hparams["mode"]=="batchnorm":
            x = self.layers[-1](self.batch_norm[-1](x))
        else:
            x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, m = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds   = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)
        accuracy = (preds == y_class).float().mean()
        # logs
        self.log('epoch', torch.tensor(self.current_epoch, dtype=torch.float32), on_epoch=True, logger=True)
        self.log('train_loss', loss, on_epoch=False, on_step=True, logger=True, batch_size=self.hparams.batch_size)
        self.log('train_accuracy', accuracy, on_epoch=False, on_step=True, logger=True, batch_size=self.hparams.batch_size)   
        self.log('train_loss_epoch', loss, on_epoch=True, on_step=False, logger=False)
        self.log('train_accuracy_epoch', accuracy, on_epoch=True, on_step=False, logger=False)
        # self.log_gradients()
        # self.log_output(m,F.softmax(logits, dim=1),preds)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds   = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)
        accuracy = (preds == y_class).float().mean()
        rps = ut.avg_rps(logits.cpu().numpy(),y.cpu().numpy())

        # self.log_output(m,F.softmax(logits, dim=1),preds)
        self.log('val_loss', loss, logger=False, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('val_accuracy', accuracy, logger=False, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('val_rps', rps, logger=False, prog_bar=True, batch_size=self.hparams.batch_size)
        # self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, logger=True)

    def on_train_epoch_end(self):
        # Registrar métricas en TensorBoard al final de cada epoch
        train_loss = self.trainer.callback_metrics.get("train_loss_epoch")
        train_accuracy = self.trainer.callback_metrics.get("train_accuracy_epoch")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_accuracy = self.trainer.callback_metrics.get("val_accuracy")
        rps = self.trainer.callback_metrics.get("val_rps")
        # dict metrics initialization
        # Escribir en TensorBoard
        if self.logger:
            tb_logger = self.logger.experiment  # Acceso al SummaryWriter de TensorBoard
            epoch = self.current_epoch+1
            if train_loss is not None:
                tb_logger.add_scalar("train_loss_epoch", train_loss, epoch)
                self.metrics["train_loss_epoch"] += [train_loss]
            if train_accuracy is not None:
                tb_logger.add_scalar("train_accuracy_epoch", train_accuracy, epoch)
                self.metrics["train_accuracy_epoch"] += [train_accuracy]
            if val_loss is not None:
                tb_logger.add_scalar("val_loss", val_loss, epoch)
                self.metrics["val_loss"] += [val_loss]
            if val_accuracy is not None:
                tb_logger.add_scalar("val_accuracy", val_accuracy, epoch)
                self.metrics["val_accuracy"] += [val_accuracy]
            if rps is not None:
                tb_logger.add_scalar("val_rps", rps, epoch)
                self.metrics["val_rps"] += [rps]
                self.metrics["rps"] += [rps]

    def on_train_end(self):
        eval_metric_values = self.metrics[self.hparams['eval_metric']] # Métrica monitoreada (por ejemplo, val_accuracy)
        if self.hparams['direction'].lower() == 'maximize':
            best_epoch = torch.argmax(torch.tensor(eval_metric_values)).item()
        else:
            best_epoch = torch.argmin(torch.tensor(eval_metric_values)).item()
        # get logged metrics
        train_loss = self.metrics.get("train_loss_epoch")
        train_accuracy = self.metrics.get("train_accuracy_epoch")
        val_loss = self.metrics.get("val_loss")
        val_accuracy = self.metrics.get("val_accuracy")
        rps = self.metrics.get("val_rps")
        # log hyperparams
        tb_logger = self.logger
        tb_logger.experiment.add_hparams(
            hparam_dict = dict(self.hparams),
            metric_dict = {
                'train_loss_m': train_loss[best_epoch],
                'train_accuracy_m': train_accuracy[best_epoch],
                'val_loss_m': val_loss[best_epoch],
                'val_accuracy_m': val_accuracy[best_epoch],
                'val_rps_m': rps[best_epoch],
            }
        )
        # Close Logger
        tb_logger.finalize('success')

    def log_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.global_step)

    def log_output(self,matches,probabilities,predictions):
        for i in range(len(matches)):
            match = matches[i]
            match_probabilities = probabilities[i]
            match_prediction = predictions[i]

            # Log the probabilities for each class
            for j, class_probability in enumerate(match_probabilities):
                self.logger.experiment.add_scalar("match_{}_class_{}".format(match,j), class_probability, global_step=self.global_step)

            # Log the prediction and the match id
            self.logger.experiment.add_scalar("match_{}_prediction".format(match), match_prediction, global_step=self.global_step)
            # self.logger.experiment.add_scalar("match".format(match), match, global_step=self.global_step)

    def configure_optimizers(self):
        optim_choose = self.hparams["optim"]
        if optim_choose=="adam":
            optimizer = optim.Adam(self.parameters(), 
                                   lr=self.hparams['learning_rate'], 
                                   betas=(self.hparams["b1"], self.hparams["b2"])
                                   )
        elif optim_choose=="sgd":
            optimizer = optim.SGD(self.parameters(), 
                                  lr=self.hparams['learning_rate'],
                                  momentum=self.hparams["momentum"],
                                  nesterov=self.hparams["nesterov"]
                                  )
        return optimizer

    def __get_activation(self):
        if self.hparams.activation == "relu":
            return nn.ReLU()
        elif self.hparams.activation == "selu":
            return nn.SELU()
        elif self.hparams.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=self.hparams.negative_slope)
        else:
            raise(NameError(f"Valor de activación NO Valido: {self.hparams.activation}"))
    
    def set_training_data(self,traindata:TorchData,valdata:TorchData,testdata:TorchData):
        self.datatrain  = traindata
        self.dataval    = valdata
        self.datatest   = testdata

    def train_dataloader(self):
        return DataLoader(self.datatrain, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True) # shuffle=True

    def val_dataloader(self):
        return DataLoader(self.dataval, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.datatest, batch_size=len(self.datatest), num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def get_probs_from_logits(self,logits=torch.Tensor):
        probs = F.softmax(logits[0],dim=1).numpy()
        return probs
    
    def compute_metrics(self,probs:np.ndarray):
        labels = self.datatest.label.argmax(dim=1).numpy()
        preds = probs.argmax(axis=1)
        accuracy = (labels==preds).mean()
        return accuracy
    

class MLP_wrapped(torch.nn.Module):
    def __init__(self, model: MLP):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        # with torch.no_grad():
        # logits = self.model(x)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(next(self.model.parameters()).device)
        logits = self.model(x_tensor)
        return torch.softmax(logits, dim=1).cpu().detach().numpy()