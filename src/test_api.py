# src/test_api.py
"""
Script de prueba para FastAPI - Día 6
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_api_endpoints():
    """Probar todos los endpoints de la API"""
    print("🧪 Probando Bloom API - Día 6")
    print("=" * 40)
    
    # 1. Test endpoint raíz
    print("1️⃣ Probando endpoint raíz...")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"   ✅ Status: {response.status_code}")
        print(f"   📝 Response: {response.json()['message']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 2. Test health check
    print("2️⃣ Probando health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        data = response.json()
        print(f"   ✅ Status: {data['status']}")
        print(f"   🤖 BERT: {data['models_loaded']['bert_classifier']}")
        print(f"   🤖 T5: {data['models_loaded']['t5_generator']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 3. Test análisis de texto
    print("3️⃣ Probando análisis de texto...")
    try:
        payload = {"text": "This API is working great!"}
        response = requests.post(f"{API_BASE}/analyze", json=payload)
        data = response.json()
        print(f"   📝 Texto: '{data['text']}'")
        print(f"   🎯 Sentimiento: {data['sentiment']} ({data['confidence']})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 4. Test generación de texto
    print("4️⃣ Probando generación de texto...")
    try:
        payload = {"prompt": "translate English to Spanish: Hello API", "max_length": 30}
        response = requests.post(f"{API_BASE}/generate", json=payload)
        data = response.json()
        print(f"   📝 Prompt: '{data['prompt']}'")
        print(f"   🤖 Generado: '{data['generated_text']}'")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 5. Test preview Bloom
    print("5️⃣ Probando preview Bloom...")
    try:
        payload = {"prompt": "machine learning concepts", "max_length": 40}
        response = requests.post(f"{API_BASE}/bloom-preview", json=payload)
        data = response.json()
        print(f"   📚 Contenido: '{data['content']}'")
        print(f"   🎓 Nivel Bloom: {data['bloom_level']}")
        print(f"   ❓ Pregunta: '{data['generated_question']}'")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 Pruebas completadas!")

if __name__ == "__main__":
    print("⏳ Esperando que la API esté lista...")
    time.sleep(2)
    test_api_endpoints()