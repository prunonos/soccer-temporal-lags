import pandas as pd
from data.owntop1 import Dataset_OwnTop1
from experiments.gb_own import GB_own
from experiments.gb_top1 import GB_top1
from experiments.gb_ratings import GB_ratings_berrar
from experiments.own_top1 import Own_top1
from experiments.tabnet_owntop1 import Tabnet_Owntop1
from experiments.tabnet_pi_ratings import Tabnet_piRatings
from experiments.tabnet_own import Tabnet_Own
from experiments.own_original import Own_Original
from data.own import Dataset_Own
from data.top1 import Dataset_top1
from data.pi_ratings import Dataset_piRatings
from data.ratings_berrar import Dataset_ratings_berrar
import utils as ut
from data_filters import apply_filters
import multiprocessing
import sys, os, time, logging, traceback
from pathlib import Path
import copy
from datetime import datetime

# Raíz del repositorio (src/..): aquí viven _pending/, _done/ y logs/
REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
for _d in (LOGS_DIR, LOGS_DIR / "models", LOGS_DIR / "metadata"):
    os.makedirs(_d, exist_ok=True)

# Configuración del logging
logging.basicConfig(
    filename=str(LOGS_DIR / 'errors.log'),  # Nombre del archivo de logs
    level=logging.ERROR,    # Nivel de log para capturar errores
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato del log
    datefmt='%Y-%m-%d %H:%M:%S'
)


def run(options,dtypes):
    
    try:
        # IMPORT RAW DATA
        rawdata = ut.read_data(options['paths']['raw_data'],dtypes=dtypes)
        # data filters configured for the 'load' stage (data.filters in the YAML)
        rawdata = apply_filters(rawdata, options.get('data', {}), stage='load', label='raw')
        # desactive CUDA if chosen
        if not options['training'].get('cuda',True): 
            print("CUDA DISALLOWED")
            os.environ["CUDA_VISIBLE_DEVICES"]=""

        pd.set_option('mode.chained_assignment', None)

        # CHOOSE EXPERIMENT
        if options["experiment"]=="own":
            pass
        elif options["experiment"]=='lgbm_top1':
            print(f"INFO - Running LightGBM top1: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_top1(rawdata,options)
            model_type = 'xgboost'
            GB_top1(model_type,dataset,options)
        elif options["experiment"]=='tabnet_own':
            print(f"INFO - Running Tabnet Own: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_Own(rawdata,options)
            Tabnet_Own(dataset,options)
        elif options["experiment"]=='tabnet_paper':
            print(f"INFO - Running Tabnet Paper: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_piRatings(rawdata,options)
            Tabnet_piRatings(dataset,options)
        elif options["experiment"]=='own_original':
            print(f"INFO - Running Own experiment: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_Own(rawdata,options)
            Own_Original(dataset,options)
        elif options["experiment"]=='own_top1':
            print(f"INFO - Running Own model with top1 data with DL nets: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_OwnTop1(rawdata,options)
            Own_top1(dataset,options)
        elif options["experiment"]=='tabnet_owntop1':
            print(f"INFO - Running Tanet model with top1 data: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_OwnTop1(rawdata,options)
            Tabnet_Owntop1(dataset,options)
        elif options["experiment"]=='ratings_berrar':
            print(f"INFO - Running LightGBM Ratings Berrar (paper): {options['experiment_id']} - {options['description']}")
            dataset = Dataset_ratings_berrar(rawdata,options)
            GB_ratings_berrar(dataset,options)
        elif options["experiment"]=='lgbm_owntop1':
            print(f"INFO - Running LightGBM Own top1: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_OwnTop1(rawdata,options)
            model_type = 'lightgbm'
            GB_top1(model_type,dataset,options)
        elif options["experiment"]=='lgbm_own':
            print(f"INFO - Running LightGBM Own: {options['experiment_id']} - {options['description']}")
            dataset = Dataset_Own(rawdata,options)
            model_type = 'lightgbm'
            GB_own(model_type,dataset,options)

        print(f"INFO - Finished Succesfully: {options['experiment_id']} - {options['description']}")
    
    except KeyboardInterrupt:
        print("Interrupted by user:",options['experiment_id'])
    except Exception as e:
        # Capturar la traza completa del error
        error_message = f"Experiment ID: {options['experiment_id']}\n"
        error_message += traceback.format_exc()
        # print(error_message)
        logging.error(error_message)  # Guardar la traza en el archivo de logs  
        raise e             

def get_project_root() -> Path:
    """
    Detecta dinámicamente la raíz del proyecto.
    Prioridad:
    1. Variable de entorno TFG_ROOT si existe y es un directorio válido.
    2. Búsqueda ascendente desde la ubicación de este script hasta encontrar la carpeta 'datasets'.
    3. Fallback: raíz del repositorio (src/..).
    """
    env_root = os.environ.get("TFG_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()

    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        if (parent / "datasets").is_dir():
            return parent.resolve()

    return REPO_ROOT


def process_yaml(queue_path, done_path, queue_name, root=None):
    if root is None:
        root = get_project_root()
    root = Path(root).resolve()
    queue_path = str(queue_path)
    done_path = str(done_path)

    cont = 0
    experiments_failed = []
    while cont < 5:
        cont += 1
        # Obtener el siguiente YAML de la cola
        pending_configs = sorted(os.listdir(queue_path))
        pending_configs_with_priority = []

        for config in pending_configs:
            if config.endswith(".yaml"):
                options = ut.load_yaml(os.path.join(queue_path, config))
                # Verificar si el YAML pertenece a la cola actual
                if options.get('queue', "default") == queue_name and options["enable"] and options['experiment_id'] not in experiments_failed:
                    priority = options.get('priority', 2)  # Prioridad predeterminada: 2
                    pending_configs_with_priority.append((priority, config, options))

        # Ordenar por prioridad descendente
        pending_configs_with_priority.sort(reverse=True, key=lambda x: x[0])
        cont = 0 if len(pending_configs_with_priority) else cont

        # for _, config, options in pending_configs_with_priority:
        if len(pending_configs_with_priority):
            config, options = pending_configs_with_priority[0][1:]  # cogemos el mas prioritario de la cola

            # Validar la configuración del experimento antes de proceder
            is_valid, validation_errors = ut.validate_experiment_config(options, os.path.join(queue_path, config))
            if not is_valid:
                error_header = f"Experiment ID: {options.get('experiment_id', config)} - YAML Validation Failed"
                print(f"\n[ERROR] {error_header}:")
                for err in validation_errors:
                    print(f"   * {err}")
                log_msg = f"{error_header}:\n" + "\n".join([f"  - {err}" for err in validation_errors])
                logging.error(log_msg)
                experiments_failed.append(options.get('experiment_id', config))
                time.sleep(2)
                continue

            # Guardamos copia limpia de las rutas relativas originales en formato portable POSIX
            options["_relative_paths"] = {
                k: str(v).replace("\\", "/") for k, v in options["paths"].items() if k != "root"
            }


            # Guardamos metadata de ejecución para trazabilidad y reproducibilidad
            options["execution_info"] = {
                "root_used": str(root).replace("\\", "/"),
                "platform": sys.platform,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Cargar dtypes resolviendo la ruta con el root dinámico
            dtypes_rel = options['paths']['dtypes_path'].lstrip("./").lstrip("/").lstrip(".\\").lstrip("\\")
            dtypes = ut.load_json(str((root / dtypes_rel).resolve()))

            # Resolver rutas para el runtime (absolutas para datasets, relativas locales para logger)
            for key in list(options["paths"].keys()):
                if key == "root":
                    del options["paths"]["root"]
                    continue
                val = str(options["paths"][key])
                if val.startswith('./') or val.startswith('.\\'):
                    options["paths"][key] = val.replace('./', '/').replace('.\\', '/')
                else:
                    clean_rel = val.lstrip("/\\")
                    options["paths"][key] = str((root / clean_rel).resolve())

            # Ejecutar el proceso
            process = multiprocessing.Process(target=run, args=(options, dtypes))
            process.start()
            process.join()
            
            # Mover el YAML a la carpeta de completados si se ejecutó con éxito
            if process.exitcode == 0:
                os.replace(os.path.join(queue_path, config), os.path.join(done_path, config))
            else:
                experiments_failed.append(options['experiment_id'])
                time.sleep(60)  # esperamos 1 minuto para empezar el siguiente experimento

        if len(pending_configs_with_priority) < 2: time.sleep(10) # si ya no quedan experimentos esperamos 10" para volver a mapear el directorio
    print("Experimentos fallidos:", experiments_failed)

if __name__=="__main__":
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)

    # Las rutas relativas de los modelos (./logs/models, logs/metadata...) se resuelven contra el CWD:
    # fijamos el CWD en la raíz del repo para poder lanzar el script desde cualquier carpeta
    os.chdir(REPO_ROOT)
    root = get_project_root()
    framework_dir = REPO_ROOT

    env_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    env = f"/{env_arg}" if env_arg and not env_arg.startswith("/") else env_arg

    # Resolución dinámica de carpetas de cola (_pending y _done)
    pending_folder = f"_pending{env}" if (framework_dir / f"_pending{env}").is_dir() else "_pending"
    done_folder = f"_done{env}" if (framework_dir / f"_done{env}").is_dir() else "_done"
    path_pending = str(framework_dir / pending_folder) + ("/" if sys.platform != 'win32' else "\\")
    path_done = str(framework_dir / done_folder) + ("/" if sys.platform != 'win32' else "\\")

    queue_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    print("Project root -----> ", str(root))
    print("Running env ------> ", env if env else 'prd')
    print("Running queue ----> ", queue_name)
    print("Reading path -----> ", path_pending)
    # Spawn multiple processes for training or evaluation
    process_yaml(path_pending, path_done, queue_name, root)
