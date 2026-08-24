# Dockerfile pour l'application Streamlit de prévision énergétique
# Ce fichier Dockerfile est utilisé pour créer une image Docker contenant l'application Streamlit et toutes ses dépendances. L'image résultante peut être déployée sur n'importe quel environnement compatible avec Docker.
# On choisit l'image "slim" pour réduire la taille de l'image finale, car elle contient moins de bibliothèques et d'outils préinstallés.
FROM python:3.10

#  Définition du dossier de travail à l'intérieur du conteneur
WORKDIR /app
# Installation de libgomp1 pour LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

#  Copie du fichier des dépendances Python dans le conteneur
COPY requirements.txt .

# Installation des dépendances Python
# On utilise l'option --no-cache-dir pour éviter de stocker les fichiers temporaires de pip, ce qui réduit la taille finale de l'image Docker.
RUN pip install --no-cache-dir -r requirements.txt

#  Copie du reste de l'application dans le conteneur
COPY . .

#  Exposition du port sur lequel Streamlit va tourner    
EXPOSE 8501
#  Commande par défaut pour lancer l'application Streamlit
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]