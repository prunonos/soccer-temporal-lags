from typing import List
from data.base import Dataset
import utils as ut
import numpy as np
from pandas import DataFrame, Series

COLS_H      = ['matchId','Div','Date','season','Weekday','id_H','HomeTeam','FTHG','FTAG','FTR','HS','HST','HF','HC', 'HY','HR','HO','HHW']
COLS_A      = ['matchId','Div','Date','season','Weekday','id_A','AwayTeam','FTAG','FTHG','FTR','AS','AST','AF','AC', 'AY','AR','AO','AHW']
COLS_AUX    = ['matchId','Div','Date','season','Weekday','idTeam','Team','FTG','FTG_rival','FTR','S','ST','F','C', 'Y','R','O','HW']
KEY_COLS   = ['matchId','Div','Date','season','Weekday']
COL_LABEL   = "label"
COLS_ORDER  = KEY_COLS + ['idTeam_H','idTeam_A','Team_H','Team_A','FTG_H','FTG_A','FTR_H',COL_LABEL]
METADATA = KEY_COLS + ["_id","HomeTeam","AwayTeam","FTHG","FTAG"]
RENAMES = {"Team_H":"HomeTeam","Team_A":"AwayTeam","FTG_H": "FTHG","FTG_A": "FTAG"}
STATIC_FEATURES = ['Weekday']

class Dataset_Own(Dataset):
    def __init__(s,data,options):
        super().__init__(data,options)
        
        # GUARDA EN s. LOS OBJETOS HASTA PREPARE DATA (que devuelve objetos externos)

    def process_data(s):
        split_data = ut.split_data_side(s.data,COLS_H,COLS_A,COLS_AUX)
        s.features = []
        if 'lags' not in s.options.get('exclusions',[]):
            s._set_features()
            s.data = s.__compute_lags(split_data)
            s.data = s._create_label("FTR",COL_LABEL)
            s.data = ut.merge_sides(s.data, KEY_COLS, [*COLS_ORDER,*s.features],label='FTR_H')
            # s.data = s.data.rename(columns={'FTR_H','FTR'})
            s.data = ut.rename_columns(s.data,RENAMES)
            s.data = s._drop_nan_values(s.data,s.features)
            s.data = s._set_index(s.data)
            # if s.options.get("balance",False): s.data = s._balance_classes(s.data)
            # if s.options.get('factor',False): s.data.loc[:,"dif_result"] = ut.dif_result(s.data)
        else:
            s.data = ut.empty_df()

    def __compute_lags(s,data):
        """
        Computes all the lag features.
        """
        s.log_print("Computing lags...")
        return ut.compute_lags(data,s.options["lags"],
                                    s.options["min_samples"],
                                    s.options["col_window"],
                                    s.options["col_group"],
                                    s.options["aggregations"]
                            )
        
    def apply_feature_transformation(s,dims:int,*data,method=""):
        data, feats = ut.transform_data(dims,s.features,*data,
                                 method=method,
                                 cols_meta=METADATA,
                                 col_label=COL_LABEL
                                )
        s.features = feats
        return data

    def _set_features(s):
        static_features = [ feat for feat in STATIC_FEATURES if feat in s.options.get('features',[]) ]
        s.features = np.concatenate((static_features,ut.get_features(s.options["lags"],s.options["aggregations"])))    