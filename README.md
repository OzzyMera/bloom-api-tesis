# 🌸 Bloom API - Generación de Preguntas y Retroalimentación Adaptativa con IA

Proyecto de tesis para construir una API educativa inteligente basada en modelos de lenguaje (T5, BERT) y la Taxonomía de Bloom.

---

## 🚀 Objetivo

Desarrollar una API capaz de:
- Generar preguntas educativas alineadas con niveles de la Taxonomía de Bloom (usando T5)
- Analizar respuestas de estudiantes para clasificar nivel de comprensión (usando BERT)
- Generar retroalimentación adaptativa en función del desempeño

---

## 📦 Tecnologías utilizadas

- [FastAPI](https://fastapi.tiangolo.com/) — Framework web ligero y rápido para APIs
- [Transformers](https://huggingface.co/transformers/) — Modelos de lenguaje preentrenados
- [PyTorch](https://pytorch.org/) — Backend de aprendizaje profundo
- [Conda](https://www.anaconda.com/) — Gestión de entornos virtuales

---

### ✅ Estado actual (Día 6):
- FastAPI funcionando en puerto 8000
- BERT classifier integrado (99.95% accuracy)
- T5 generator integrado
- Documentación automática: http://localhost:8000/docs

### 🚀 Cómo probar:
```bash
conda activate bloom-api
python src/api/main.py  # Terminal 1
python src/test_api.py  # Terminal 2
