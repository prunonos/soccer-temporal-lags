import os, sys, yaml, json, random, re
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, normalize, minmax_scale
# from tensorboard.backend.event_processing import event_accumulator
random.seed(0)

if sys.platform=='win32':
    slash = '\\'
else: 
    slash = '/'

pd.set_option('mode.chained_assignment', None)

def read_data(path,dtypes=None):
    return pd.read_csv(path,sep=';',decimal=',',parse_dates=['Date'],
                       date_format="%d/%m/%Y",dtype=dtypes)

def save_dataframe(df,path,name_of_file):
    df.to_csv(f"{path}_{name_of_file}.csv",decimal=',',sep=';',)

def load_yaml(path):
    yaml_file = open(path, 'r')
    yaml_content = yaml.safe_load(yaml_file)
    return yaml_content

def load_json(path):
    """
    Reads a JSON file and returns a dictionary.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The dictionary representation of the JSON content.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_yaml(data, path):
    with open(path, 'w') as yaml_file:
        yaml.safe_dump(data, yaml_file)

def save_dict_as_json(dictionary, file_path):
    # with open(file_path+'/hparams.json', 'w') as json_file:
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file)    

def filter_list(expr:str, lista:list):
    # Utilizar una comprensión de lista para filtrar los elementos que coinciden con la expresión regular
    coincidencias = [elemento for elemento in lista if re.match(expr, elemento)]
    return coincidencias

def getPoints(df,scored,received,new_col="Points"):
    df.loc[:,new_col] = 1
    df.loc[df[scored]>df[received], new_col] = 3
    df.loc[df[scored]<df[received], new_col] = 0
    return df

def split_data_side(data,cols_h,cols_a,cols_aux):
    df_H = data[cols_h]
    df_H.columns = cols_aux
    df_H.loc[:,'Side'] = 0
    df_H = getPoints(df_H,"FTG","FTG_rival")

    df_A = data[cols_a]
    df_A.columns = cols_aux
    df_A.loc[:,'Side'] = 1
    df_A = getPoints(df_A,"FTG","FTG_rival")

    df = pd.concat([df_H,df_A])
    df.sort_values(['Date','matchId'],ascending=True,inplace=True)
    return df

def empty_df() -> pd.DataFrame:
    return pd.DataFrame()

def compute_lag(df:pd.DataFrame,lag:int,min_samples:int,col_window:str,cols_group:List[str],
                                aggregations:Dict[str,List[str]],cols_agg=False) -> pd.DataFrame:
    if cols_agg==False: cols_agg = [ k+"_"+str(lag) for k in aggregations.keys() ]
    df_agg = (df.groupby(cols_group)
                .rolling(window=lag,min_periods=min_samples,on=col_window,closed='left')
                .agg(aggregations)
    )
    df_agg.columns = cols_agg
    return df_agg

def compute_lags(df,lags,min_samples,col_window,cols_group,aggregations,cols_agg=False):
    old_df = df
    for lag,min_ in zip(lags,min_samples):
        df_agg = compute_lag(old_df,lag,min_,col_window,cols_group,aggregations,cols_agg=False)
        df = df.merge(df_agg.reset_index(), on=[col_window,*cols_group], how='left')
    df_count = df.groupby(['idTeam','Date']).count()
    assert(len(df_count[df_count.matchId>1]) == 0)
    del(df_count)
    return df

def get_features(lags,aggregations):
    features = []
    for agg in aggregations.keys():
        for lag in lags:
            for side in ["H","A"]:
                features.append(agg+'_'+lag+"_"+side)
    features = np.array(features).reshape(-1)
    return features

def ensamble_data(data:List[pd.DataFrame],col_on:str) -> pd.DataFrame:
    if not data: return pd.DataFrame()
    for i,d in enumerate(data):
        if len(d): break 
    df = data[i].reset_index(drop=True)
    for d in data[i+1:]:
        d = d.reset_index(drop=True)
        if len(d): df = df.merge(d,on=col_on,how='left').dropna()
    return df

def _create_label(df,col,newCol="label"):
    df.loc[df[col]=='D',newCol] = 0
    df.loc[df[col]=='H',newCol] = 1
    df.loc[df[col]=='A',newCol] = 2
    return df

def merge_sides(df:pd.DataFrame,group_on,col_order,label='label'):
    mask_home = df.Side==0
    mask_away = df.Side==1
    data_merged = df[mask_home].merge(df[mask_away], on=group_on, suffixes=("_H","_A"))
    if label:
        data_merged.loc[:,"label"] = data_merged[label]
        _create_label(data_merged,"label","label")
    data_merged = data_merged[drop_duplicates(col_order)]
    data_merged.drop(label,axis=1,errors='ignore')
    return data_merged

def drop_duplicates(x):
  return list(dict.fromkeys(x))

def rename_columns(df: pd.DataFrame,renames: dict):
    return df.rename(columns=renames)

def split_random(df,last_digits=[5,9]):
    df.loc[:,"last_digit"] = df.matchId % 10
    mask_test = False
    for dig in last_digits:
        mask_test   = mask_test | (df.last_digit==dig)
    data_train  = df[~mask_test]#.reset_index()
    data_test   = df[mask_test]#.reset_index()
    print("INFO: Train is ", len(data_train)/len(df) )
    df["split"] = "Train"
    df.loc[mask_test,"split"] = "Test"
    df = df.drop(columns="last_digit")
    return df, data_train, data_test

def split_sequential(df,date):
    data_train = df[df.Date<date].sort_values("Date",ascending=True)#.reset_index()
    data_val  = df[df.Date>=date].sort_values("Date",ascending=True)#.reset_index() 
    df["split"] = "Train"
    df.loc[df.Date>=date,"split"] = "Val"
    return df, data_train, data_val

def balance_dataset(df,col='FTR_H'):
    if col=='label':
        home_label, draw_label, away_label = 1,0,2
    elif 'FTR' in col:
        home_label, draw_label, away_label = 'H', 'D', 'A'
    home_count = sum(df[col]==home_label)
    draws = df[df[col]==draw_label]
    draws_sample = draws.sample(home_count-len(draws),replace=True,random_state=0)
    draws = pd.concat([draws,draws_sample])
    away_wins = df[df[col]==away_label]
    away_sample = away_wins.sample(home_count-len(away_wins),replace=True,random_state=0)
    away_wins = pd.concat([away_wins,away_sample])
    assert len(away_wins) == home_count == len(draws)
    return pd.concat([df[df[col]==home_label],draws,away_wins]).sort_values("Date",ascending=True).reset_index(drop=True)

def dif_result(df:pd.DataFrame):
        return (df['FTHG']-df['FTAG']).abs() + 1  #.astype(float)

def factor_labels(labels,dif_result,factor,scaler="normalizer"):
    labels = labels * dif_result
    if scaler=="normalizer": return normalize(labels,axis=0)*factor
    elif scaler=='minmax': return minmax_scale(labels,axis=0,feature_range=(0,factor))
    elif scaler=='maxabs': return minmax_scale(labels,axis=0,feature_range=(-factor,factor))
    else: return labels

def select_scaler(scaler):
    if scaler=='normalizer':
        return Normalizer
    if scaler=='minmax':
        return MinMaxScaler
    if scaler=='maxabs':
        return MaxAbsScaler
    raise Exception("No se ha pasado un scaler válido.")

def get_event_logs(experiment, version):
    return sorted([ ev for ev in os.listdir(os.getcwd()+slash+"logs"+slash+experiment+slash+version) if ev.startswith("events.out.tfevents.") ])

# def save_outputs(experiment,datatrain):
#     eventos_experiment = pd.DataFrame([])
#     get_match = lambda x: x.split('_')[1] if 'match' in x else -1

#     for i,version in enumerate(os.listdir(os.getcwd()+slash+"logs"+slash+experiment)):
#             if version.endswith(".csv"): continue
#             events_files = get_event_logs(experiment, version)
#             version_id = version.split('_')[-1]
#             print(version, version_id)
#             for event_file in events_files[-1:]:
#                 event_acc = event_accumulator.EventAccumulator(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+event_file)
#                 event_acc.Reload()

#                 matches = list(map(get_match, event_acc.Tags()['scalars']))
#                 matches = np.array(matches).astype(int)
#                 matches = np.unique(matches[matches>-1])

#                 prob_draw, prob_home, prob_away, preds, epoch, matchesId = [], [], [], [], [], []

#                 # iterar sobre cada partido
#                 for m in matches:
#                         # añadir cada evento (class 0-1-2 y prediction) a su correspondiente lista
#                         matchesId.extend([ m for _ in event_acc.Scalars(f'match_{m}_class_0') ])
#                         prob_draw.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_class_0')])
#                         prob_home.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_class_1')])
#                         prob_away.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_class_2')])
#                         preds.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_prediction')])
#                         epoch.extend([ item.step for item in  event_acc.Scalars(f'match_{m}_class_0')])

#                 # crear dataframe
#                 eventos = pd.DataFrame({
#                                         "matchId": matchesId,
#                                         "epoch":epoch,
#                                         "prob_draw":prob_draw,
#                                         "prob_home":prob_home,
#                                         "prob_away":prob_away,
#                                         "predictions":preds
#                                         })
#                 eventos["version"] = version_id
#                 eventos_experiment = pd.concat([eventos_experiment,eventos])

#     cols = ['matchId','Div','Date','season','idTeam_H','idTeam_A','Team_H','Team_A','FTG_H','FTG_A','label','split']
#     datatrain = datatrain[cols]
#     df_output = datatrain.merge(eventos_experiment,on='matchId',how='left')
#     path_save = f"{os.getcwd()}{slash}logs{slash}{experiment}{slash}{experiment}_outputs.csv"
#     df_output.to_csv(path_save,sep=';',decimal=',',encoding='utf-8',date_format="%d/%m/%Y",index=False)
#     print(f"INFO {experiment} - Output saved in {path_save}")

# # unir dataframe con la info de los partidos

# def save_metrics(experiment):
#     metrics_experiment = pd.DataFrame([])
#     metrics = ["test_loss","test_accuracy","test_rps"]

#     for i,version in enumerate(os.listdir(os.getcwd()+slash+"logs"+slash+experiment)):
#             if version.endswith(".csv"): continue
#             events_files = get_event_logs(experiment, version)

#             test_metrics = event_accumulator.EventAccumulator(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+events_files[-1])
#             test_metrics.Reload()

#             metrics = {
#                 "version": [],
#                 "test_loss": [],
#                 "test_accuracy": [],
#                 "test_rps": [],
#             }
#             version_id = version.split('_')[-1]
#             # print(version, version_id)
#             metrics["version"].append(version_id)
#             metrics["test_loss"].append(test_metrics.Scalars('test_loss')[-1].value)
#             metrics["test_accuracy"].append(test_metrics.Scalars('test_accuracy')[-1].value)
#             metrics["test_rps"].append(test_metrics.Scalars('test_rps')[-1].value)
            
#             # crear dataframe
#             metrics = pd.DataFrame(metrics)
#             metrics_experiment = pd.concat([metrics_experiment,metrics])

#     path_save = f"{os.getcwd()}{slash}logs{slash}{experiment}{slash}{experiment}_metrics.csv"
#     metrics_experiment.to_csv(path_save,sep=';',decimal=',',encoding='utf-8',date_format="%d/%m/%Y",index=False)
#     print(f"INFO {experiment} - Output saved in {path_save}")

def delete_event_logs(experiment) -> None:
    for version in os.listdir(os.getcwd()+slash+"logs"+slash+experiment):
            if version.endswith(".csv"): continue
            events_files = get_event_logs(experiment, version)[-1:]
            for file in events_files:
                if os.path.exists(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+file):
                    os.remove(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+file)
                else:
                    print(f"ERROR {experiment} - The file {file} does not exist.")

def ranked_probability_score(outcomes, labels):
    _, r = labels.shape  # n: número de registros, m: número de dimensiones
    # Calcula el RPS
    rps = 1 / (r - 1) * np.sum((np.sum(labels[:, :i], axis=1) - np.sum(outcomes[:, :i], axis=1))**2 for i in range(1, r))  
    return rps

def avg_rps(logits,actual):
    # nos aseguramos que los dos arrays estan one-hot encoded
    if len(logits.shape)<2 or logits.shape[1]!=3: logits = np.eye(logits.max()+1)[logits]
    if len(actual.shape)<2 or actual.shape[1]!=3: actual = np.eye(actual.max()+1)[actual]
    rps = ranked_probability_score(logits,actual)
    return np.mean(rps).round(4)  


#########################
# FEATURE ENGINEERING

def transform_data(dims: int, cols_feats: List[str], *args,
                        method: str = "",
                        cols_meta: List[str] = [],
                        col_label: str = 'label'
                    ) -> Tuple[List[pd.DataFrame],List[str]]: 
    res = []
    if dims<len(cols_feats):
        if method=="anova":
            train = args[0]    # train set to compute the new dimensional space 
            metadata = train[cols_meta]
            train_labels = train[col_label]
            train, cols_feats = anova(train[cols_feats],dims,train_labels)
            train.loc[:,col_label] = train_labels
            train = train.join(metadata)
            res.append(train)
            for setdata in args[1:]: # we apply the anova to the remaining sets
                newset = setdata[[*cols_meta,*cols_feats,col_label]]
                res.append(newset)
        elif method=="pca":
            data = [_set[cols_feats] for _set in args]
            (data,cols_feats) = pca(dims,*data)
            for d_feats, d_meta in zip(data,args):
                d_meta  = d_meta.loc[:,[*cols_meta,col_label]]
                d_whole = pd.concat([d_feats,d_meta.reset_index(drop=True)],axis=1).set_index(d_meta.index)
                res.append(d_whole) 
        else:
            res = args
    else:
        res = args
    return res, cols_feats

def pca(dims:int, *args: pd.DataFrame) -> Tuple[List[pd.DataFrame],np.ndarray]:
    res = []
    traindata = args[0]
    new_features = np.arange(dims)+1

    pca = PCA(n_components=dims,random_state=0).fit(traindata)
    for setdata in args:
        newset = pd.DataFrame(data=pca.transform(setdata),columns=new_features)
        res.append(newset)
    return res, new_features

def anova(data,dims,labels):
    old_data = data.copy()
    data = data.dropna()
    labels = labels.loc[data.index]
    X_mean = data.mean(axis=0).to_numpy()
    X_norm = data / X_mean
    features = X_norm.columns

    filter  = feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=dims)
    filter.fit(X_norm,labels)
    mask_new_feat = filter.get_support()
    data = old_data.loc[:,mask_new_feat]
    features = features[mask_new_feat]
    return data, features


def validate_experiment_config(options: dict, raw_file_path: str = None) -> Tuple[bool, List[str]]:
    """
    Validates experiment configuration dictionary loaded from YAML.
    Performs checks on mandatory fields, metric/direction consistency,
    parameter alignment, boolean typos, and date validity.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []

    if not isinstance(options, dict):
        return False, ["Configuration must be a valid dictionary."]

    # 1. Mandatory top-level sections
    for section in ["experiment", "experiment_id", "data", "paths", "training"]:
        if section not in options:
            errors.append(f"Missing mandatory section '{section}'")

    data_cfg = options.get("data", {})
    train_cfg = options.get("training", {})
    paths_cfg = options.get("paths", {})
    exp_id = str(options.get("experiment_id", ""))

    # 2. Select alignment between data and training
    data_select = data_cfg.get("select") if isinstance(data_cfg, dict) else None
    train_select = train_cfg.get("select") if isinstance(train_cfg, dict) else None
    if data_select is not None and train_select is not None:
        if data_select != train_select:
            errors.append(f"Inconsistent 'select': data.select ({data_select}) != training.select ({train_select})")

    # 3. Direction vs. Metric alignment
    if isinstance(train_cfg, dict):
        eval_metric = train_cfg.get("eval_metric", "")
        primary_metric = str(eval_metric[-1] if isinstance(eval_metric, list) else eval_metric).lower()
        direction = str(train_cfg.get("direction", "")).lower()

        if "accuracy" in primary_metric and direction != "maximize":
            errors.append(f"Metric '{primary_metric}' requires direction 'maximize', but found '{direction}'")
        elif "rps" in primary_metric and direction != "minimize":
            errors.append(f"Metric '{primary_metric}' requires direction 'minimize', but found '{direction}'")

    # 4. Normal (non-batch) experiment thresholds
    is_batch = bool(data_select) or bool(train_select) or "batch_" in exp_id or "test_" in exp_id
    if not is_batch and isinstance(train_cfg, dict):
        epochs = train_cfg.get("max_epochs", 0)
        iters = train_cfg.get("iterations", 0)
        if isinstance(epochs, (int, float)) and epochs <= 10:
            errors.append(f"Non-batch experiment '{exp_id}' must have max_epochs > 10, found {epochs}")
        if isinstance(iters, (int, float)) and iters <= 2:
            errors.append(f"Non-batch experiment '{exp_id}' must have iterations > 2, found {iters}")

    # 5. Boolean typos detection (e.g. 'Fals', 'Tru', 'flase', 'ture')
    BOOLEAN_TYPOS = {"tru", "ture", "trur", "fals", "flase", "fale", "flas"}

    def check_boolean_typos(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_boolean_typos(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                check_boolean_typos(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            clean_str = obj.strip().lower()
            if clean_str in BOOLEAN_TYPOS:
                errors.append(f"Suspected boolean typo '{obj}' in field '{prefix}'. Expected 'true' or 'false'")

    check_boolean_typos(options)

    # 6. Date validation
    if isinstance(data_cfg, dict):
        val_date_str = data_cfg.get("val_date")
        test_date_str = data_cfg.get("test_date")
        test_end_date_str = data_cfg.get("test_end_date")
        test_range = data_cfg.get("test_range")

        if val_date_str and test_date_str:
            try:
                val_dt = pd.to_datetime(val_date_str)
                test_dt = pd.to_datetime(test_date_str)
                if val_dt >= test_dt:
                    errors.append(f"val_date ({val_date_str}) must be earlier than test_date ({test_date_str})")
            except Exception as e:
                errors.append(f"Error parsing dates (val_date: {val_date_str}, test_date: {test_date_str}): {str(e)}")

        if test_date_str and test_end_date_str:
            try:
                test_dt = pd.to_datetime(test_date_str)
                test_end_dt = pd.to_datetime(test_end_date_str)
                if test_dt > test_end_dt:
                    errors.append(f"test_date ({test_date_str}) must be <= test_end_date ({test_end_date_str})")
            except Exception as e:
                errors.append(f"Error parsing test_end_date ({test_end_date_str}): {str(e)}")

        if test_range is not None and str(test_range).lower() not in ["season", "year", "none", ""]:
            errors.append(f"Invalid test_range '{test_range}'. Allowed options: 'season', 'year'")

        # 7. Data filters (data.filters / legacy data.leagues)
        try:
            from data_filters import normalize_filters
            normalize_filters(data_cfg)
        except ValueError as e:
            errors.append(f"Invalid data filters: {e}")

    return len(errors) == 0, errors

