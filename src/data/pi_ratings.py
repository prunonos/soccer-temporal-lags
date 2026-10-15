from typing import Dict
import numpy as np
import pandas as pd
from data.base import Dataset
import copy, utils as ut

COLS_META   = ['matchId','Div','Date','season']
COL_LABEL   = "label"
COLS_ORDER  = [*COLS_META,"id_H","id_A","HomeTeam","AwayTeam","FTHG","FTAG",COL_LABEL]
COLS_FEATS  = ["rate_home","rate_away"]

class Dataset_piRatings(Dataset):
    def __init__(s, data, options):
        super().__init__(data, options)
        s.features = COLS_FEATS

    def process_data(s):
        s.data = s._create_label("FTR",COL_LABEL)
        s.data = s.data.loc[:,COLS_ORDER] # el label aun no existe
        s.data_pretrain = s.data[s.data.Date<s.options["init_date"]]
        s.init_rates_dict =  { t:{"rate_home":0,"rate_away":0,"rate_global":0} for t in s.data.id_H.unique() }
        s.data_pretrain = s.initialize_rates(s.data_pretrain)
        s.data_pretrain = s.initialize_columns(s.data_pretrain)
        s.c, s.b, s.lamda, s.gamma = 3, 10, 0.01, 0.6
        s.compute_pi_rates(s.data_pretrain,s.lamda,s.gamma,s.init_rates_dict)
        s.data = s.initialize_rates(s.data[s.data.Date>=s.options["init_date"]])
        s.data = s.initialize_columns(s.data)

    def asign_col(s,df,col_name,value):
        df.loc[:,col_name] = value

    def get_rate_global(self,elemH,elemA):
        return (elemH+elemA)/2

    def initialize_rates(s,data:pd.DataFrame) -> pd.DataFrame:
        for row in data.itertuples():
            idx = row.Index
            team_H = row.id_H
            team_A = row.id_A
            data.loc[idx,"rate_home"] = s.init_rates_dict[team_H]["rate_home"]
            data.loc[idx,"rate_away"] = s.init_rates_dict[team_A]["rate_away"]
        return data

    def initialize_columns(s,data:pd.DataFrame) -> pd.DataFrame:
        s.asign_col(data,"new_rate_home",0.0)
        s.asign_col(data,"new_rate_away",0.0)
        rate_global = s.get_rate_global(data.rate_home,data.rate_away)
        s.asign_col(data,"rate_global",rate_global)
        s.asign_col(data,"new_rate_global",0.0)
        s.asign_col(data,"gd_actual", data.FTHG - data.FTAG)
        s.asign_col(data,"gd_pred",0.0)
        s.asign_col(data,"error",0.0)
        return data

    def train_pi_rates(s,data:pd.DataFrame,lamda,gamma):
        s.initialize_columns(data)
        s.initialize_rates(data)
        dict_rates = copy.deepcopy(s.init_rates_dict)
        s.compute_pi_rates(data,lamda,gamma,dict_rates)

    def compute_pi_rates(s,data:pd.DataFrame,lamda,gamma,dict_rates:Dict) -> None:
        s.log_print("Computing piRates...")
        for row in data.itertuples():
            idx = row.Index
            team_H = row.id_H
            team_A = row.id_A
            data.loc[idx,"rate_home"] = dict_rates[team_H]["rate_home"]
            data.loc[idx,"rate_away"] = dict_rates[team_A]["rate_away"]
            gd_H_pred = s.b ** (abs(dict_rates[team_H]["rate_home"])/s.c) - 1
            gd_A_pred = s.b ** (abs(dict_rates[team_A]["rate_away"])/s.c) - 1
            data.loc[idx,"gd_pred"] = gd_H_pred - gd_A_pred
            error_H, error_A = s.error_func(row.gd_actual,row.gd_pred)
            data.loc[idx,"error"] = abs(error_H)

            new_rate_home_H = dict_rates[team_H]["rate_home"] + error_H * lamda
            new_rate_home_A = dict_rates[team_H]["rate_away"] + (new_rate_home_H - dict_rates[team_H]["rate_home"]) * gamma
            new_rate_away_A = dict_rates[team_A]["rate_away"] + error_A * lamda
            new_rate_away_H = dict_rates[team_A]["rate_home"] + (new_rate_away_A - dict_rates[team_A]["rate_away"]) * gamma

            dict_rates[team_H]["rate_home"] = new_rate_home_H
            dict_rates[team_H]["rate_away"] = new_rate_home_A
            dict_rates[team_H]["rate_global"] = (new_rate_home_H + new_rate_home_A) / 2
            dict_rates[team_A]["rate_away"] = new_rate_away_A
            dict_rates[team_A]["rate_home"] = new_rate_away_H
            dict_rates[team_A]["rate_global"] = (new_rate_away_H + new_rate_away_A) / 2
                    
            data.loc[idx,"new_rate_home"] = new_rate_home_H
            data.loc[idx,"new_rate_away"] = new_rate_away_A

    def error_func(s,actual,pred):
        error = abs(actual-pred)
        error = s.c * np.log10(1+error)
        error_H = -error
        error_A = -error
        if actual>pred:
                error_H = error
        if actual<pred: 
                error_A = error
        return error_H, error_A