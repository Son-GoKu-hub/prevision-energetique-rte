import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def time_series_split(
    df: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données chronologiquement."""
    if "date_heure" not in df.columns:
        raise ValueError("La colonne 'date_heure' est absente.")

    df = df.copy()
    df["date_heure"] = pd.to_datetime(df["date_heure"])

    logging.info("Split temporel des données à la date : %s", split_date)

    train_df = df[df["date_heure"] < split_date].copy()
    test_df = df[df["date_heure"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Le train set ou le test set est vide.")

    logging.info("Taille du Train set : %d lignes.", len(train_df))
    logging.info("Taille du Test set : %d lignes.", len(test_df))

    return train_df, test_df


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[LGBMRegressor, pd.DataFrame]:
    """Entraîne LightGBM, prédit et calcule les métriques."""
    if target_col not in train_df.columns:
        raise ValueError(f"La cible '{target_col}' est absente des données.")

    train_df = train_df.dropna().copy()
    test_df = test_df.dropna().copy()

    excluded_columns = {"date_heure", "date", target_col}
    features = [
        column for column in train_df.columns
        if column not in excluded_columns
    ]

    if not features:
        raise ValueError("Aucune variable explicative disponible.")

    missing_features = set(features) - set(test_df.columns)
    if missing_features:
        raise ValueError(
            f"Colonnes absentes du test set : {sorted(missing_features)}"
        )

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

# On convertit les colonnes catégorielles en type 'category' pour LightGBM
    X_train = X_train.select_dtypes(include=["number", "bool"])
    X_test = X_test.select_dtypes(include=["number", "bool"])

    logging.info("Entraînement du modèle LightGBM...")

    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    logging.info("Génération des prédictions sur le Test set...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    non_zero_values = y_test != 0
    if non_zero_values.any():
        mape = np.mean(
            np.abs(
                (y_test[non_zero_values] - predictions[non_zero_values])
                / y_test[non_zero_values]
            )
        ) * 100
    else:
        mape = float("nan")

    logging.info("--- RÉSULTATS DU MODÈLE ---")
    logging.info("MAE  : %.2f MW", mae)
    logging.info("MAPE : %.2f %%", mape)

    test_results = test_df.copy()
    test_results["prediction"] = predictions

    return model, test_results


def analyze_errors_by_segment(test_results: pd.DataFrame) -> None:
    """Analyse les erreurs par heure et type de jour."""
    required_columns = {
        "heure",
        "est_weekend",
        "reel",
        "prediction",
    }

    missing_columns = required_columns - set(test_results.columns)
    if missing_columns:
        logging.warning(
            "Analyse ignorée. Colonnes absentes : %s",
            sorted(missing_columns),
        )
        return

    df_eval = test_results.copy()
    df_eval["residu"] = df_eval["reel"] - df_eval["prediction"]

    denominator = df_eval["reel"].replace(0, np.nan)
    df_eval["APE"] = (
        df_eval["residu"].abs() / denominator.abs() * 100
    )

    logging.info("--- ANALYSE DES ERREURS ---")
    logging.info(
        "Biais moyen : %.2f MW",
        df_eval["residu"].mean(),
    )
    logging.info(
        "Écart-type des résidus : %.2f MW",
        df_eval["residu"].std(),
    )

    mape_by_hour = df_eval.groupby("heure")["APE"].mean().dropna()

    if not mape_by_hour.empty:
        worst_hour = mape_by_hour.idxmax()
        best_hour = mape_by_hour.idxmin()

        logging.info(
            "Heure la plus difficile : %sh (MAPE : %.2f %%)",
            worst_hour,
            mape_by_hour.loc[worst_hour],
        )
        logging.info(
            "Heure la plus facile : %sh (MAPE : %.2f %%)",
            best_hour,
            mape_by_hour.loc[best_hour],
        )

    weekday_mape = df_eval.loc[df_eval["est_weekend"] == 0, "APE"].mean()
    weekend_mape = df_eval.loc[df_eval["est_weekend"] == 1, "APE"].mean()

    logging.info("MAPE en semaine : %.2f %%", weekday_mape)
    logging.info("MAPE le week-end : %.2f %%", weekend_mape)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    data_path = (
        root_dir / "data" / "processed" / "features_eco2mix.parquet"
    )
    model_path = root_dir / "src" / "lgbm_model.pkl"
    target = "consommation"

    try:
        df = pd.read_parquet(data_path)
        df.columns = df.columns.str.replace(" ", "_", regex=False)
        df["date_heure"] = pd.to_datetime(df["date_heure"])

        split_date = (
            df["date_heure"].max() - pd.Timedelta(days=30)
        ).strftime("%Y-%m-%d")

        train_df, test_df = time_series_split(df, split_date)

        model, test_results = train_and_evaluate(
            train_df,
            test_df,
            target,
        )

        test_results["reel"] = test_results[target]
        analyze_errors_by_segment(test_results)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        logging.info("Modèle sauvegardé dans : %s", model_path)

    except Exception:
        logging.exception("Erreur lors de la modélisation")