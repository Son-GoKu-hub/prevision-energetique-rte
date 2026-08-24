import pandas as pd
import logging
from pathlib import Path
from jours_feries_france import JoursFeries

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def create_calendar_features(df: pd.DataFrame, datetime_col: str = 'date_heure') -> pd.DataFrame:
    """
    Extrait les composantes calendaires et identifie les jours fériés.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant l'historique.
        datetime_col (str): Le nom de la colonne temporelle.
        
    Returns:
        pd.DataFrame: Le DataFrame enrichi des features calendaires.
    """
    logging.info("Création des variables calendaires...")
    df = df.copy()
    df.columns = df.columns.str.replace(' ', '_')
    
    # Sécurité : s'assurer que la colonne est bien un objet datetime Pandas
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    
    # 1. Extraction des cycles naturels
    df['heure'] = df[datetime_col].dt.hour
    df['jour_semaine'] = df[datetime_col].dt.dayofweek # Lundi = 0, Dimanche = 6
    df['mois'] = df[datetime_col].dt.month
    df['jour_annee'] = df[datetime_col].dt.dayofyear
    
    # 2. Identification des week-ends (0 ou 1)
    df['est_weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)
    
    # 3. Identification des jours fériés
    # On identifie toutes les années présentes dans notre dataset
    annees_presentes = df[datetime_col].dt.year.unique()
    jours_feries_dates = []
    
    for annee in annees_presentes:
        # Récupère un dictionnaire des jours fériés de l'année
        jf = JoursFeries.for_year(annee)
        jours_feries_dates.extend(jf.values())
        
    # Conversion en objets 'date' pour la comparaison
    jours_feries_dates = pd.to_datetime(jours_feries_dates).date
    
    # On compare la partie 'date' de notre colonne datetime avec la liste des jours fériés
    df['est_ferie'] = df[datetime_col].dt.date.isin(jours_feries_dates).astype(int)
    
    return df

def create_lag_features(df: pd.DataFrame, target_col: str = 'consommation', freq_hours: float = 0.5) -> pd.DataFrame:
    """
    Crée des variables retardées (lags) pour capter l'inertie de la consommation.
    
    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        target_col (str): La variable que l'on cherche à prédire plus tard.
        freq_hours (float): Le pas de temps du dataset en heures (0.5 pour 30 min).
        
    Returns:
        pd.DataFrame: Le DataFrame enrichi des lags.
    """
    logging.info("Création des variables de lags temporels...")
    df = df.copy()
    
    # Règle d'or absolue en Time Series : TOUJOURS trier chronologiquement avant un shift
    # Si le dataset est mélangé, le lag n'aura aucun sens mathématique.
    df = df.sort_values('date_heure').reset_index(drop=True)
    
    # Calcul du nombre de lignes à décaler selon la fréquence des données
    # RTE éCO2mix est généralement au pas de 30 minutes (0.5 heure).
    # Donc 24h = 48 lignes de décalage.
    rows_per_day = int(24 / freq_hours)
    
    # Création des décalages (J-1 et J-7)
    df['conso_lag_24h'] = df[target_col].shift(rows_per_day)
    df['conso_lag_1_semaine'] = df[target_col].shift(rows_per_day * 7)
    
    return df

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "eco2mix_2023_2025.parquet"
    PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "features_eco2mix.parquet"
    
    try:
        # Chargement des données brutes
        df_raw = pd.read_parquet(RAW_DATA_PATH)
        
        # Opendatasoft nomme souvent la colonne cible 'consommation' (à vérifier selon ton extraction)
        TARGET = 'consommation' 
        
        # Application du pipeline de transformations
        df_features = create_calendar_features(df_raw, datetime_col='date_heure')
        df_features = create_lag_features(df_features, target_col=TARGET, freq_hours=0.5)
        
        # Sauvegarde
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(PROCESSED_DATA_PATH, index=False)
        logging.info(f"Feature Engineering terminé. Données sauvegardées dans {PROCESSED_DATA_PATH}")
        
    except Exception as e:
        logging.error(f"Erreur lors du Feature Engineering : {e}")