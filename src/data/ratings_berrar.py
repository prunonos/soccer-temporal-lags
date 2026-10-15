import math
from typing import Dict
import pandas as pd
from data.base import Dataset
import copy
from sklearn.metrics import mean_squared_error as error_func


COLS_META   = ['matchId','Div','Date','season']
COL_LABEL   = "label"
COLS_ORDER  = [*COLS_META,"id_H","id_A","HomeTeam","AwayTeam","FTHG","FTAG",COL_LABEL]
COLS_FEATS  = ["rate_home_att","rate_home_def","rate_away_att","rate_away_def"]

class Dataset_ratings_berrar(Dataset):
    def __init__(s, data, options):
        super().__init__(data, options)
        s.features = s.options.get("ratings",dict()).get("features",COLS_FEATS)

    def process_data(s):
        s.data = s._create_label("FTR",COL_LABEL)
        s.data = s.data.loc[:,COLS_ORDER] # el label aun no existe
        # we get pre train data for rating initialization
        s.data_pretrain = s.data[s.data.Date<s.options["init_date"]]
        s.init_rates_dict =  { t:{"rate_home_att":0.,"rate_home_def":0.,"rate_away_att":0.,"rate_away_def":0.} 
                                for t in s.data.HomeTeam.unique() }
        s.data_pretrain = s.initialize_rates(s.data_pretrain)
        s.data_pretrain = s.initialize_columns(s.data_pretrain)
        # compute init ratings
        s.compute_pi_rates(s.data_pretrain,s.init_rates_dict)
        # prepare ratings for training
        s.data = s.initialize_rates(s.data[s.data.Date>=s.options["init_date"]])
        s.data = s.initialize_columns(s.data)

    def asign_col(s,df,col_name,value):
        df.loc[:,col_name] = value

    def goal_match_team(s,team_attack,rival_defensive,alpha,beta,phi):
            return alpha / ( 1 + math.exp(-beta*(team_attack+rival_defensive) - phi) )

    def update_rating(s,current_rate, weight, truth, pred):
            return current_rate + weight*(truth-pred)
    
    def initialize_rates(s,data:pd.DataFrame) -> pd.DataFrame:
        for row in data.itertuples():
            idx = row.Index
            team_H = row.HomeTeam
            team_A = row.AwayTeam
            data.loc[idx,"rate_home_att"] = s.init_rates_dict[team_H]["rate_home_att"]
            data.loc[idx,"rate_home_def"] = s.init_rates_dict[team_H]["rate_home_def"]
            data.loc[idx,"rate_away_att"] = s.init_rates_dict[team_A]["rate_away_att"]
            data.loc[idx,"rate_away_def"] = s.init_rates_dict[team_A]["rate_away_def"]
        return data

    def initialize_columns(s,data:pd.DataFrame) -> pd.DataFrame:
        # s.asign_col(data,"new_rate_home_att",0.0)
        # s.asign_col(data,"new_rate_away_att",0.0)
        # s.asign_col(data,"new_rate_home_def",0.0)
        # s.asign_col(data,"new_rate_away_def",0.0)
        s.asign_col(data,"error",0.0)
        return data

    def train_pi_rates(s,data:pd.DataFrame, alphaH=5, alphaA=5, betaH=1, betaA=1, phiH=0, phiA=0, 
            weightHomeAttack=0.5, weightHomeDef=0.5, weightAwayAttack=0.5, weightAwayDef=0.5
    ) -> None:
        s.initialize_columns(data)
        s.initialize_rates(data)
        dict_rates = copy.deepcopy(s.init_rates_dict)
        s.compute_pi_rates(data, dict_rates, alphaH, alphaA, betaH, betaA, phiH, phiA, 
            weightHomeAttack, weightHomeDef, weightAwayAttack, weightAwayDef)

    def compute_pi_rates(
        s, data: pd.DataFrame, dict_rates: Dict[str,Dict[str,float]], alphaH=5, alphaA=5, betaH=1, betaA=1, phiH=0, phiA=0, 
            weightHomeAttack=0.5, weightHomeDef=0.5, weightAwayAttack=0.5, weightAwayDef=0.5
    ) -> None:
        s.log_print("Computing piRates...")
        for row in data.itertuples():
            idx = row.Index
            team_H = row.HomeTeam
            team_A = row.AwayTeam
            # inicializamos ratings para predecir el partido actual
            rate_home_att = dict_rates[team_H].get("rate_home_att",-1.)
            rate_away_att = dict_rates[team_A].get("rate_away_att",-1.)
            rate_home_def = dict_rates[team_H].get("rate_home_def",-1.)
            rate_away_def = dict_rates[team_A].get("rate_away_def",-1.)

            data.loc[idx,"rate_home_att"] = rate_home_att
            data.loc[idx,"rate_away_att"] = rate_away_att
            data.loc[idx,"rate_home_def"] = rate_home_def
            data.loc[idx,"rate_away_def"] = rate_away_def

            # calculamos los goles para cada equipo
            gd_H_pred = s.goal_match_team(rate_home_att, rate_away_def, alphaH, betaH, phiH)
            gd_A_pred = s.goal_match_team(rate_away_att, rate_home_def, alphaA, betaA, phiA)
            # calculamos diferencia de goles y error
            error = error_func([row.FTHG,row.FTAG],[gd_H_pred,gd_A_pred])
            data.loc[idx,"error"] = abs(error)

            # actualizamos ratings en base al error
            new_rate_home_att = s.update_rating(rate_home_att, weightHomeAttack, row.FTHG, gd_H_pred)
            new_rate_home_def = s.update_rating(rate_home_def, weightHomeDef, row.FTAG, gd_A_pred)
            new_rate_away_att = s.update_rating(rate_away_att, weightAwayAttack, row.FTAG, gd_A_pred)
            new_rate_away_def = s.update_rating(rate_away_def, weightAwayDef, row.FTHG, gd_H_pred)

            # actualizamos diccionario auxiliar para 
            dict_rates[team_H]["rate_home_att"] = new_rate_home_att
            dict_rates[team_A]["rate_away_att"] = new_rate_away_att
            dict_rates[team_H]["rate_home_def"] = new_rate_home_def
            dict_rates[team_A]["rate_away_def"] = new_rate_away_def
                
            data.loc[idx,"pFTHG"] = round(gd_H_pred)
            data.loc[idx,"pFTAG"] = round(gd_A_pred)
            data.loc[idx,"lFTHG"] = gd_H_pred
            data.loc[idx,"lFTAG"] = gd_A_pred