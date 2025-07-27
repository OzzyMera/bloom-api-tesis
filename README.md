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

## ⚙️ Setup rápido del entorno

```bash
# Clonar el repositorio
git clone https://github.com/OzzyMera/bloom-api-tesis.git
cd bloom-api-tesis

# Crear entorno con conda
conda create -n bloom-api python=3.9
conda activate bloom-api

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar API
uvicorn src.main:app --reload
