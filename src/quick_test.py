# src/quick_test.py
"""
Verificación rápida - Día 5
"""

def quick_test():
    print("🧪 Verificación rápida del setup:")
    
    try:
        from transformers import pipeline
        print("✅ Transformers importado correctamente")
        
        # Prueba súper rápida
        classifier = pipeline("sentiment-analysis")
        result = classifier("I love this setup!")
        print(f"✅ Modelo funciona: {result}")
        
        print("\n🎉 Todo funcionando correctamente!")
        print("🚀 Ejecuta: python src/models/model_loader.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Ejecuta: pip install -r requirements.txt")

if __name__ == "__main__":
    quick_test()