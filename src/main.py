# src/main.py

from fastapi import FastAPI
from src.models.model_loader import load_bert_classifier, load_t5_generator

app = FastAPI(
    title="Bloom API",
    version="0.1.0",
    description="API para generación de preguntas y retroalimentación educativa usando modelos de lenguaje"
)

# Cargar modelos al iniciar
bert_model = load_bert_classifier()
t5_model = load_t5_generator()

@app.get("/")
def root():
    return {"message": "🌸 Bloom API funcionando correctamente"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "bert_model_loaded": bert_model is not None,
        "t5_model_loaded": t5_model is not None
    }
