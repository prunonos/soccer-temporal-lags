import argparse
from collections import OrderedDict
import os
import re
import yaml

def generar_nombre_archivo(data,suffix=''):
    experiment = data["experiment"]
    balanced = "_bal" if data["data"].get("balance",False) else ""
    sampling = "_ss" if  data["data"].get("sample")=="sequential" else "_rs"
    batch = "batch_" if data["data"].get("select",False) else "" # batch now at the beggining of the filename
    advanced = '_adv' if data["data"].get("features",False) and 'top1' in experiment else ""
    rps_arg = data["training"].get("eval_metric",'-')
    rps = '_rps' if rps_arg=='rps' or ('rps' in rps_arg[-1]) else "" 
    supervised, aggregations, factor = "", "", ""
    if experiment.startswith("tabnet"):
        supervised = "_sup" if data["training"]["unsupervised_training"] else "_unsup"
        if experiment.endswith("own") or experiment.endswith("owntop1"):
            aggregations = "_comp" if "S" in data["data"].get("aggregations",{}).keys() else "_simp"
    elif experiment.startswith("own"):
        aggregations = "_comp" if "S" in data["data"].get("aggregations",{}) else "_simp"
        factor_flag = data["data"].get("factor",False)
        if factor_flag=='mandatory': factor = '_facman'
        elif factor_flag or factor_flag=='optional': factor = "" # --> old '_fac'
        else: factor = "_nofac"
    elif experiment=='lgbm_own':
        aggregations = "_comp" if "S" in data["data"].get("aggregations",{}) else "_simp"
    else:
        name = data["experiment_id"]
    
    suffix = suffix if (suffix.startswith('_')) or (not suffix) else '_'+suffix
    name = batch + experiment + advanced + aggregations + balanced + factor + sampling + supervised + rps + suffix
    # data["experiment_id"] = name
    # print(f"New name: {name}\n")
    return name

FILTER_STAGES = ("load", "split")

def parse_filter(spec):
    """
    'NOMBRE:V1,V2,...[:STAGE]' -> {'name': NOMBRE, 'values': [V1, V2, ...], 'stage': STAGE}
    Ej: 'leagues:E0,SP1,D1,I1,F1'  |  'leagues:E0,SP1:load'   (stage por defecto: split)
    """
    parts = spec.split(':')
    if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(f"Filtro mal formado '{spec}'. Formato: NOMBRE:V1,V2,...[:STAGE]")
    name, values = parts[0].strip(), [v.strip() for v in parts[1].split(',') if v.strip()]
    stage = parts[2].strip() if len(parts) == 3 else 'split'
    if stage not in FILTER_STAGES:
        raise argparse.ArgumentTypeError(f"Stage '{stage}' no valido en '{spec}'. Opciones: {FILTER_STAGES}")
    if not values:
        raise argparse.ArgumentTypeError(f"El filtro '{spec}' no tiene valores")
    return {'name': name, 'values': values, 'stage': stage}

def aplicar_filtros(data_cfg, filtros, clear=False):
    """
    Añade (o reemplaza, si ya existe uno con el mismo nombre) filtros en data.filters.
    Si se añade un filtro 'leagues', el atajo antiguo data.leagues se elimina para evitar ambigüedad.
    """
    actuales = [] if clear else list(data_cfg.get('filters') or [])
    for f in filtros:
        actuales = [a for a in actuales if a.get('name') != f['name']] + [f]
        if f['name'] == 'leagues' and 'leagues' in data_cfg:
            print(f"  data.leagues={data_cfg['leagues']} eliminado (sustituido por data.filters/leagues)")
            del data_cfg['leagues']
    if actuales:
        data_cfg['filters'] = actuales
    elif 'filters' in data_cfg:
        del data_cfg['filters']
    return data_cfg

def validar_filtros(data_cfg):
    """Valida con src/data_filters.py del framework (si se puede importar)."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
        from data_filters import normalize_filters
    except Exception:
        return
    normalize_filters(data_cfg)  # lanza ValueError si la config no es valida

def modificar_yaml(archivo, campos_a_modificar, replace=True, drop=[], adding=[], suffix='', name=False,
                   filtros=[], clear_filters=False):
    # for archivo in archivos:
    # Cargar el archivo YAML
    with open(archivo, 'r') as file:
        data = yaml.safe_load(file)

    # Crear un OrderedDict para mantener el orden de los campos
    nuevo_data = OrderedDict()

    # Copiamos data en el OrderedDict
    for campo, valor in data.items():
        nuevo_data[campo] = valor

    for d in drop:
        nuevo_data = remove_key_path(nuevo_data,d)

    # Modificar campos segun los valores proporcionados
    for campo, valor in campos_a_modificar.items():
        # Usar un valor predeterminado si el campo no esta presente en el YAML original
        if isinstance(nuevo_data.get(campo,None),dict):
            nuevo_data[campo].update(valor)
        elif isinstance(nuevo_data.get(campo,None),list) and (campo in adding):
            nuevo_data[campo] += valor
        else:
            nuevo_data[campo] = valor
        print(archivo,campo,valor)

    # Filtros de datos (data.filters)
    if filtros or clear_filters:
        nuevo_data['data'] = aplicar_filtros(nuevo_data.get('data') or {}, filtros, clear=clear_filters)
        validar_filtros(nuevo_data['data'])
        print(archivo, 'data.filters', nuevo_data['data'].get('filters', []))

    if name=='true': name = archivo.replace('.yaml','').replace('.yml','')
    elif name=='false' or not name: name = generar_nombre_archivo(nuevo_data,suffix)
    else: name = nuevo_data["experiment"] + '_' + name
    nuevo_data["experiment_id"] = name
    print(f"New name: {name}")
    
    # Guardar el archivo modificado
    name = name if name.endswith('yaml') else name+'.yaml'
    with open(name, 'w') as filew:
        yaml.dump(dict(nuevo_data), filew, default_flow_style=False)
    
    if replace and name!=archivo:
        os.remove(archivo)

    return name
   
def parse_elements(string):
    pattern = r'\[([^\[\]]*)\]'
    match = re.search(pattern, string)
    if match:
        elements_str = match.group(1)
        elements = [elem.strip() for elem in elements_str.split(',')]
        parsed_elements = []
        for elem in elements:
            if elem.isdigit():
                parsed_elem = int(elem)
            else:
                try:
                    parsed_elem = float(elem)
                except ValueError:
                    parsed_elem = elem
            parsed_elements.append(parsed_elem)
        return parsed_elements
    else:
        return string

def update_nested_dict(d, keys, value):
    if keys=='': return d
    key_list = keys.split(".")
    current_dict = d
    for key in key_list[:-1]:
        current_dict = current_dict.setdefault(key, {})
    if value.isdigit(): value = int(value)
    elif value.isdecimal(): value = float(value)
    elif type(value)==str and value.lower()=='true': value = True
    elif type(value)==str and value.lower()=='false': value = False
    value = parse_elements(value) if isinstance(value,str) else value
    current_dict[key_list[-1]] = value
    return d

def remove_key_path(d, keys):
    key_list = keys.split(".")
    current_dict = d
    for key in key_list[:-1]:
        current_dict = current_dict.get(key, {})
    
    if key_list[-1] in current_dict:
        del current_dict[key_list[-1]]
    return d


if __name__ == "__main__":
    # Lista de archivos YAML a modificar
    parser = argparse.ArgumentParser(description='Modificar archivos YAML.')
    parser.add_argument('-r','--replace',action="store_true",default=False)
    parser.add_argument('archivos', nargs='+', help='Archivos YAML a modificar')
    parser.add_argument('-d','--drop', type=str, nargs='*',default=[]) # si es un campo anidado tal que asi "A" : { "B": 1 } -> A.B
    parser.add_argument('-a','--add', type=str, nargs='*',default=[])  # si es un campo tipo lista se aÃ±ade el valor a la lista
    parser.add_argument('-f','--fields',nargs='*',default=[])
    parser.add_argument('-v','--values',nargs='*',default=[])
    parser.add_argument('-s','--suffix',type=str, default='')
    parser.add_argument('-n','--name',type=str,default=False) # True = keep same name | False = create new name | "string" = new name
    parser.add_argument('-t','--test',action="store_true",default=False)
    # filtros de datos -> data.filters. Repetible. Formato NOMBRE:V1,V2,...[:STAGE] (STAGE: split|load, por defecto split)
    # ej: --filter leagues:E0,SP1,D1,I1,F1   |   --filter leagues:E0,SP1:load
    parser.add_argument('--filter',dest='filters',type=parse_filter,action='append',default=[])
    parser.add_argument('--clear-filters',action="store_true",default=False) # elimina los filtros existentes antes de añadir
    args = parser.parse_args()

    # Campos y valores a modificar
    campos_modificar = {
        "enable":True,
        # "data": {"select":1000},
        # "training": {
        #     "select": 1000,
        #     "timeout":3700,
        #     "max_epochs": 120,
        #     "iterations": 5000
        # }
    }

    if args.test:
        campos_modificar['training'] = {}
        campos_modificar['training']['select'] = 64
        campos_modificar['training']['batch_size'] = [16,32]
        campos_modificar['training']['iterations'] = 2
        campos_modificar['training']['max_epochs'] = 2
        campos_modificar['training']['timeout'] = 15
        campos_modificar['queue'] = 'test'
        campos_modificar['data'] = {}
        campos_modificar['data']['select'] = campos_modificar['training']['select']
        
    keys = args.fields
    values = args.values
    for key,value in zip(keys,values):
        campos_modificar = update_nested_dict(campos_modificar,key,value)
    print(campos_modificar)
    # Modificar archivos YAML
    print('Archivos a modificar:', args.archivos)
    for archivo in args.archivos:
        modificar_yaml(archivo, campos_modificar, replace=args.replace, drop=args.drop,
                    adding=args.add, suffix=args.suffix, name=args.name,
                    filtros=args.filters, clear_filters=args.clear_filters)
