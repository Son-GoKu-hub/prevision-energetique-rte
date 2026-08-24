import pandas as pd
import requests
import logging
from pathlib import Path
from requests.exceptions import HTTPError, RequestException

# Configuration du logger pour le suivi en production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_rte_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extrait les données de consommation électrique depuis l'API ODRE (éCO2mix).
    
    Args:
        start_date (str): Date de début au format 'YYYY-MM-DD'.
        end_date (str): Date de fin au format 'YYYY-MM-DD'.
        
    Returns:
        pd.DataFrame: DataFrame contenant les données brutes.
    """
    logging.info(f"Début de l'extraction des données du {start_date} au {end_date}...")
    
    # URL de l'API d'export (ODS v2.1) qui permet de contourner la limite de pagination de 10 000 lignes
    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv"
    
    # Paramètres de la requête avec un filtre temporel (clause WHERE)
    params = {
        "where": f"date_heure >= '{start_date}' AND date_heure <= '{end_date}'",
        "delimiter": ";"
    }
    
    try:
        # Appel API avec un timeout pour éviter de bloquer le programme indéfiniment
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Lève une exception si le statut HTTP n'est pas 200 
        
        # Opendatasoft exporte un CSV, nous le lisons dans Pandas
        # StringIO est utilisé car on lit une chaîne de caractères depuis la mémoire
        from io import StringIO
        csv_data = StringIO(response.text)
        
        df = pd.read_csv(csv_data, sep=";")
        logging.info(f"Extraction réussie : {df.shape[0]} lignes récupérées.")
        return df
        
    except HTTPError as http_err:
        logging.error(f"Erreur HTTP lors de l'appel API : {http_err}")
        raise
    except RequestException as req_err:
        logging.error(f"Erreur de connexion/requête : {req_err}")
        raise
    except Exception as e:
        logging.error(f"Erreur inattendue : {e}")
        raise

def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Sauvegarde le DataFrame au format Parquet.
    
    Args:
        df (pd.DataFrame): Le DataFrame à sauvegarder.
        filepath (Path): Le chemin complet de sauvegarde.
    """
    try:
        # Création du dossier s'il n'existe pas
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde au format Parquet
        df.to_parquet(filepath, index=False)
        logging.info(f"Données sauvegardées avec succès dans {filepath}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde : {e}")
        raise

if __name__ == "__main__":
    # Définition des chemins avec pathlib (bonne pratique vs os.path)
    # Le script étant dans src/, le dossier racine est le parent de src/
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "eco2mix_2023_2025.parquet"
    
    # 3 dernières années pleines (par exemple)
    START = "2023-01-01"
    END = "2025-12-31"
    
    # Exécution du pipeline d'ingestion
    df_rte = fetch_rte_data(START, END)
    save_data(df_rte, RAW_DATA_PATH)