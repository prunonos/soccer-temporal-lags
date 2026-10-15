from typing import Dict, Tuple
from pandas import DataFrame
from data.own import Dataset_Own
from data.top1 import Dataset_top1
import utils as ut
import pandas as pd

KEY_COLS   = ['matchId','Div','Date','season']
METADATA    = ['matchId','Div','Date','season','id_H','id_A','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
COL_LABEL   = "label"

class Dataset_OwnTop1(Dataset_Own,Dataset_top1):
    def __init__(self, data: DataFrame, options: Dict):
        super().__init__(data, options)
        self.data_own  = Dataset_Own(data,options)
        self.data_top1 = Dataset_top1(data,options)

    def process_data(self):
        self.data_own.process_data()
        self.data_top1.process_data()
        self.feature = self._set_features()

    def _set_features(self):
        self.features = list(self.data_own.features) + self.data_top1.features

    def ensamble_data(self,lamda:float,gamma:float) -> DataFrame:
        # Hacemos tres casos porque `pi_ratings` y `lags` son los DFs que contienen la metadata, por lo que tenemos que hacer un 3r caso
        if 'lags' not in self.exclusions:
            self.data = ut.ensamble_data([self.data_own.data,
                                          self.data_top1.init_data],
                                          "matchId")    
        else:  
            self._create_label('FTR',COL_LABEL) # tenemos que añadir el label porque solo se usa en esta version sin pi_ratings
            self.data = ut.ensamble_data([self.data[[*METADATA,COL_LABEL]],self.data_top1.init_data],"matchId") # self.init_data
        if 'pi_ratings' not in self.exclusions:
            trial_piratings = self.data_top1.train_pi_rates(lamda,gamma)
            self.data = ut.ensamble_data([self.data,
                                          trial_piratings.data[["matchId","rate_home","rate_away"]]], 
                                          "matchId")

        return self.data
