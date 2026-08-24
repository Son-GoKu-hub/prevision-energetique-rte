import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# --- Configuration de la page ---
st.set_page_config(
    page_title="Prévision Énergétique (RTE)",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

# --- Définition des chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "features_eco2mix.parquet"
MODEL_PATH = ROOT_DIR / "src" / "lgbm_model.pkl"

# --- Fonctions de chargement avec mise en cache ---
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    # Alignement du nom des colonnes avec l'entraînement
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    # Conversion de la date pour être sûr du format
    df['date_heure'] = pd.to_datetime(df['date_heure'])
    return df.dropna().reset_index(drop=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# --- Interface Utilisateur ---
st.title("Prévision de la Consommation Électrique Française")
st.markdown("basée sur l'historique de RTE.")

try:
    with st.spinner('Chargement des données et du modèle...'):
        df = load_data()
        model = load_model()

    st.subheader("Filtres Temporels")
    
    # Création d'une colonne temporaire juste pour le slider
    df['date_seule'] = df['date_heure'].dt.date
    dates = df['date_seule'].unique()
    
    start_date, end_date = st.select_slider(
        "Sélectionnez la période à observer :",
        options=dates,
        value=(dates[-30], dates[-1]) 
    )

    mask = (df['date_seule'] >= start_date) & (df['date_seule'] <= end_date)
    df_filtered = df.loc[mask].copy()

    if not df_filtered.empty:
        
        # =====================================================================
        # LA CORRECTION EST ICI : On filtre strictement sur les nombres et booléens
        # et on retire la colonne cible ('consommation') et les dates.
        # =====================================================================
        X = df_filtered.drop(columns=['date_heure', 'consommation', 'date_seule'], errors='ignore')
        X = X.select_dtypes(include=['number', 'bool'])
        
        y_real = df_filtered['consommation']
        
        # Prédiction
        y_pred = model.predict(X)
        df_filtered['prediction'] = y_pred

        # KPIs
        mape = (abs(y_real - y_pred) / y_real).mean() * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Consommation Moyenne (Période)", f"{int(y_real.mean()):,} MW")
        col2.metric("Erreur (MAPE) sur la Période", f"{mape:.2f} %")

        # Graphique
        st.subheader("Comparaison : Réel vs Prédiction")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_filtered['date_heure'], y=df_filtered['consommation'], 
            mode='lines', name='Consommation Réelle', line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_filtered['date_heure'], y=df_filtered['prediction'], 
            mode='lines', name='Prédiction LightGBM', line=dict(color='#ff7f0e', width=2, dash='dot')
        ))
        
        fig.update_layout(
            xaxis_title="Date et Heure", yaxis_title="Consommation (MW)",
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Aucune donnée disponible pour cette période.")

except Exception as e:
    st.error(f"Une erreur est survenue lors du chargement de l'application : {e}")