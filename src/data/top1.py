import copy
from typing import Dict, List, Tuple
import numpy as np
from pandas import DataFrame
from data.base import Dataset
import utils as ut
import pandas as pd
from data.pi_ratings import Dataset_piRatings as pirates

COLS_H      = ['matchId','Div','Date','season','id_H','HomeTeam','FTHG','FTAG']
COLS_A      = ['matchId','Div','Date','season','id_A','AwayTeam','FTAG','FTHG']
COLS_AUX    = ['matchId','Div','Date','season','idTeam','Team','FTG','FTG_rival']
METADATA    = ['matchId','Div','Date','season','id_H','id_A','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
COLS_FEATS  = ["Draw","Win","FTG_mean","FTG_std","FTG_rival_mean","FTG_rival_std"]
COL_LABEL   = "label"

class Dataset_top1(Dataset):
    def __init__(self, data: DataFrame, options: Dict):
        super().__init__(data, options)
        self.exclusions: List = self.options.get('exclusions',[])

    def process_data(self):
        self.features = self._set_features()
        dfs_dict = self.initialize_data(self.data)
        # save some meta columns from data and the FTR (label)
        # self.metadata = self.data[COLS_META]
        # split data in Sides
        data_split = ut.split_data_side(self.data,COLS_H,COLS_A,COLS_AUX)
        # set Points as one-hot-encoding to compute percentages
        data_split[["Loss","Draw","Win"]] = pd.get_dummies(data_split.Points)
        # Compute lags
        if 'long_term' not in self.exclusions: 
            dfs_dict['long_term']  = self.compute_longterm_features(data_split)
        if 'short_term' not in self.exclusions:
            dfs_dict['short_term'] = self.compute_shortterm_features(data_split)
        # Ensamble all data but pi-ratings (to be trained in every trial)
        self.init_data = ut.ensamble_data([dfs_dict[key] for key in dfs_dict.keys()],"matchId")

    def compute_longterm_features(self,df:pd.DataFrame):
        # compute the lag-features for the last 2 years with a minimum of 5 matches.
        aggs = {"Draw":["mean"],"Win":["mean"],"FTG":["mean","std"],"FTG_rival":["mean","std"]}
        long_term = ut.compute_lag(df.sort_values("Date"),"730D",5,'Date',["idTeam","Team","Side"],aggs,COLS_FEATS)
        long_term = long_term.reset_index().merge(df[['Date','idTeam','matchId']],on=['Date','idTeam'])
        # merge both sides in a united dataframe
        feats = [ [f+s for f in COLS_FEATS] for s in ['_H','_A']  ]
        feats = [ *feats[0],*feats[1] ]
        long_term = ut.merge_sides(long_term,["matchId"],col_order=["matchId",*feats],label=False)
        long_term.columns = ['matchId'] + [ "lt_"+f for f in feats ]
        return long_term

    def compute_shortterm_features(self,df:pd.DataFrame):
        # compute the lag-features for the last 2 years with a minimum of 5 matches.
        aggs = {"Draw":["mean"],"Win":["mean"],"FTG":["mean","std"],"FTG_rival":["mean","std"]}
        short_term = ut.compute_lag(df.sort_values("Date"),5,5,'Date',["idTeam","Team","season"],aggs,COLS_FEATS)
        # add Side column to the short_term feats.
        short_term = short_term.reset_index().merge(df[['Date','idTeam','Side','matchId']],on=['Date','idTeam'])
        # merge both sides in a united dataframe
        feats = [ [f+s for f in COLS_FEATS] for s in ['_H','_A']  ]
        feats = [ *feats[0],*feats[1] ]
        short_term = ut.merge_sides(short_term,["matchId"],col_order=["matchId",*feats],label=False)
        short_term.columns = ['matchId'] + [ "st_"+f for f in feats ]
        return short_term

    def initialize_data(self,data:pd.DataFrame) -> Dict[str,DataFrame]:
        """
        We initialize the three datasets:
        - Pi-ratings
        - Pagerank
        - Match importance features
        """
        dfs_dict = {}
        if 'pi_ratings' not in self.exclusions:
            self.log_print("Initializing PI-ratings...")
            options = {"data": self.options, "experiment_id":self.exp_id, "paths":self.paths}
            self.init_piratings = pirates(data,options)
            self.init_piratings.process_data() # NOT INCLUDED on dict because it is trained in each Trial
        if 'page_rank' not in self.exclusions:
            self.log_print("Loading pagerank data...")
            dfs_dict['page_rank'] = pd.read_csv(self.paths['page_rank'],sep=';',decimal=',')
        if 'page_rank_lags' not in self.exclusions:
            self.log_print("Loading pagerank lags version data...")
            page_rank = pd.read_csv(self.paths['page_rank_lags'],sep=';',decimal=',')
            feats = ut.filter_list(r"pagerank_\d+",self.options.get('features',page_rank.columns[1:])) # en caso que no se pasen las features cogemos todas excepto el match
            dfs_dict['page_rank'] = page_rank[['matchId',*feats]]
        if 'match_importance' not in self.exclusions:
            self.log_print("Loading match_importance data...")
            dfs_dict['match_importance'] = pd.read_csv(self.paths['match_importance'],sep=';',decimal=',')
        return dfs_dict

    def _set_features(self):
        feats = []
        if ('long_term' not in self.exclusions): 
            feats_lt = [ ['lt_'+f+s for f in COLS_FEATS] for s in ['_H','_A']  ]
            feats_lt = np.array(feats_lt).reshape(-1).tolist()
            feats = feats + feats_lt
        if ('short_term' not in self.exclusions): 
            feats_st = [ ['st_'+f+s for f in COLS_FEATS] for s in ['_H','_A']  ]
            feats_st = np.array(feats_st).reshape(-1).tolist()
            feats = feats + feats_st
        if 'pi_ratings' not in self.exclusions: 
            feats_pi = ["rate_home","rate_away"]
            feats = feats + feats_pi
        if 'page_rank' not in self.exclusions: 
            feats_pr = ["pagerank_H","pagerank_A"]         
            feats = feats + feats_pr
        if 'page_rank_lags' not in self.exclusions: 
            feats_prl = ut.filter_list(r"pagerank_\d+",self.options.get('features',[]))
            feats = feats + feats_prl
        if 'match_importance' not in self.exclusions: 
            feats_mi = ut.filter_list(r"(top|down)\d+_(A|H)",self.options.get('features',[]))
            feats_mi = [ [f+s for f in ['top1', 'top2', 'top3','top4', 'top5', 'down1', 'down2', 'down3', 'down4', 'down5']] for s in ['_H','_A']  ] if not(len(feats_mi)) else feats_mi
            feats_mi = [*np.array(feats_mi).reshape(-1),"rounds"]
            feats = feats + feats_mi
        self.features = feats    
        return self.features
    
    def train_pi_rates(self,lamda:float,gamma:float) -> pirates:
        copy_piratings = copy.deepcopy(self.init_piratings)
        copy_piratings.train_pi_rates(copy_piratings.data,lamda,gamma)
        return copy_piratings
    
    def ensamble_data(self,lamda:float,gamma:float) -> pd.DataFrame:
        if 'pi_ratings' not in self.exclusions:
            trial_piratings = self.train_pi_rates(lamda,gamma)
            self.data = ut.ensamble_data([trial_piratings.data,self.init_data],"matchId")
        else:
            self._create_label('FTR',COL_LABEL) # tenemos que añadir el label porque solo se usa en esta version sin pi_ratings
            self.data = ut.ensamble_data([self.data[[*METADATA,COL_LABEL]],self.init_data],"matchId") # self.init_data
        return self.data


    