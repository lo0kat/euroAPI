# euroAPI

## **Install**
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

## **Authors**
- Tony Richard
- Louis CHOMEL