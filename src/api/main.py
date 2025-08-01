# src/api/main.py
"""
Bloom API - FastAPI Principal
Día 6: Conectando modelos ML con endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Agregar src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_loader import load_bert_classifier, load_t5_generator, check_system_info

# Crear la aplicación FastAPI
app = FastAPI(
    title="Bloom API - Generación de Preguntas IA",
    description="API para generar preguntas educativas usando BERT y T5",
    version="0.1.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Variables globales para modelos (se cargan al inicio)
bert_classifier = None
t5_generator = None

# Modelos Pydantic para requests/responses
class TextAnalysisRequest(BaseModel):
    text: str
    
class TextAnalysisResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50
    
class TextGenerationResponse(BaseModel):
    prompt: str
    generated_text: str

# Eventos de startup/shutdown
@app.on_event("startup")
async def startup_event():
    """Cargar modelos al iniciar la API"""
    global bert_classifier, t5_generator
    
    print("🚀 Iniciando Bloom API...")
    print("=" * 50)
    
    # Mostrar info del sistema
    check_system_info()
    
    # Cargar modelos
    print("📦 Cargando modelos ML...")
    bert_classifier = load_bert_classifier()
    t5_generator = load_t5_generator()
    
    if bert_classifier and t5_generator:
        print("✅ API lista para recibir requests!")
    else:
        print("⚠️  Algunos modelos no se cargaron correctamente")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar"""
    print("👋 Cerrando Bloom API...")

# === ENDPOINTS BÁSICOS ===

@app.get("/")
async def root():
    """Endpoint raíz - Información básica"""
    return {
        "message": "🚀 Bloom API - Generación de Preguntas con IA",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "generate": "/generate"
        }
    }

@app.get("/health")
async def health_check():
    """Verificar estado de la API y modelos"""
    global bert_classifier, t5_generator
    
    return {
        "status": "healthy" if (bert_classifier and t5_generator) else "partial",
        "models_loaded": {
            "bert_classifier": bert_classifier is not None,
            "t5_generator": t5_generator is not None
        },
        "system_info": {
            "python_version": sys.version.split()[0],
            "api_version": "0.1.0"
        }
    }

# === ENDPOINTS DE MODELOS ML ===

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analizar sentimiento de texto con BERT"""
    global bert_classifier
    
    if not bert_classifier:
        raise HTTPException(status_code=503, detail="BERT classifier no disponible")
    
    try:
        # Usar BERT para análisis
        result = bert_classifier(request.text)
        
        return TextAnalysisResponse(
            text=request.text,
            sentiment=result[0]['label'],
            confidence=round(result[0]['score'], 4)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generar texto con T5"""
    global t5_generator
    
    if not t5_generator:
        raise HTTPException(status_code=503, detail="T5 generator no disponible")
    
    try:
        # Usar T5 para generación
        result = t5_generator(
            request.prompt, 
            max_length=request.max_length,
            num_return_sequences=1
        )
        
        return TextGenerationResponse(
            prompt=request.prompt,
            generated_text=result[0]['generated_text']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")

# === ENDPOINT DE PRUEBA ESPECÍFICO PARA BLOOM ===

@app.post("/bloom-preview")
async def bloom_preview(request: TextGenerationRequest):
    """Preview de funcionalidad Bloom (Día 6 - Demo)"""
    global t5_generator
    
    if not t5_generator:
        raise HTTPException(status_code=503, detail="T5 generator no disponible")
    
    # Simular generación de pregunta básica
    bloom_prompt = f"generate question: {request.prompt}"
    
    try:
        result = t5_generator(bloom_prompt, max_length=request.max_length)
        
        return {
            "content": request.prompt,
            "bloom_level": "recordar",  # Simulado por ahora
            "generated_question": result[0]['generated_text'],
            "note": "🚧 Preview - Funcionalidad completa en desarrollo"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# === FUNCIÓN MAIN PARA TESTING ===
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando servidor de desarrollo...")
    print("📖 Documentación: http://localhost:8000/docs")
    print("🔗 API: http://localhost:8000")
    print("❌ Para detener: Ctrl+C")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )