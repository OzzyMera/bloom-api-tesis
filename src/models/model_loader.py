from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

def check_system_info():
    """Verificar información del sistema"""
    print("🔍 Información del sistema:")
    print(f"   Python: {torch.__version__}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

def load_bert_classifier():
    """Cargar modelo BERT para clasificación"""
    print("🤖 Cargando DistilBERT (clasificador)...")
    try:
        # DistilBERT es más rápido que BERT-base
        classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ DistilBERT cargado exitosamente!")
        return classifier
    except Exception as e:
        print(f"❌ Error cargando BERT: {e}")
        return None

def load_t5_generator():
    """Cargar modelo T5 para generación"""
    print("🤖 Cargando T5-small (generador)...")
    try:
        # T5-small es perfecto para desarrollo
        generator = pipeline(
            "text2text-generation",
            model="t5-small",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ T5-small cargado exitosamente!")
        return generator
    except Exception as e:
        print(f"❌ Error cargando T5: {e}")
        return None

def test_bert_classification(classifier):
    """Probar BERT con ejemplos"""
    if not classifier:
        return
    
    print("\n🧪 Probando BERT - Análisis de sentimiento:")
    test_texts = [
        "This is a great API!",
        "I hate bugs in my code",
        "Python is awesome for ML"
    ]
    
    for text in test_texts:
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']
        print(f"   📝 '{text}' → {label} ({score:.2f})")

def test_t5_generation(generator):
    """Probar T5 con ejemplos"""
    if not generator:
        return
    
    print("\n🧪 Probando T5 - Generación de texto:")
    test_prompts = [
        "translate English to Spanish: Hello world",
        "summarize: The weather is beautiful today. The sun is shining and birds are singing.",
        "question: What is machine learning?"
    ]
    
    for prompt in test_prompts:
        result = generator(prompt, max_length=50, num_return_sequences=1)
        output = result[0]['generated_text']
        print(f"   📝 Input: '{prompt}'")
        print(f"   🤖 Output: '{output}'")
        print()

def main():
    """Función principal de prueba"""
    print("🚀 Bloom API - Día 5: Primer contacto con modelos ML")
    print("=" * 60)
    
    # Verificar sistema
    check_system_info()
    
    # Cargar modelos
    print("📦 Cargando modelos...")
    bert_classifier = load_bert_classifier()
    t5_generator = load_t5_generator()
    
    # Probar modelos
    if bert_classifier:
        test_bert_classification(bert_classifier)
    
    if t5_generator:
        test_t5_generation(t5_generator)
    
    print("\n🎯 Resumen del Día 5:")
    print(f"   ✅ BERT funcionando: {bert_classifier is not None}")
    print(f"   ✅ T5 funcionando: {t5_generator is not None}")
    
    if bert_classifier and t5_generator:
        print("\n🎉 ¡ÉXITO! Ambos modelos funcionando correctamente")
        print("🚀 Listo para Día 6: Conectar con FastAPI")
    else:
        print("\n⚠️  Revisar errores arriba")

if __name__ == "__main__":
    main()