# euroAPI
## **Pré-requis**
Il est nécessaire d'avoir une version de python entre python 3.6 et 3.9.
## **Installation**
### **Créer environnement virtuel** 
A la racine du projet :
```sh
python3 -m venv MYENV
```

Puis pour activer l'environnement:

```sh
source MYENV/bin/activate
```
## **Installation des dépendances**
A la racine du projet : 
```sh
pip3 install -r requirements.txt
```

## **Lancer l'application** 
```sh
uvicorn app.main:app --reload
```
Accèder à l'API de FastAPI sur http://localhost:8000

## Modèle
Random Forest est le modèle utilisé (profondeur de 2 par défaut). Un modèle purement statistique n'aboutit pas à des résultats satisfaisants.

## **Authors**
- Tony Richard
- Louis CHOMEL
