# Qwen3.8 streaming como Cloudera AI Workbench Application

Aplicación autocontenida para servir `Qwen/Qwen3.8-27B-FP8` mediante la API OpenAI de vLLM, incluyendo streaming SSE con `stream=true`, sobre una A100 de 80 GB.

Toda la lógica ejecutable está en `app.py`: creación del virtualenv, instalaciones pip, validación y arranque del servidor. No utiliza `requirements.txt`, `cdsw-build.sh` ni el Python global para importar vLLM.

## Crear la Application

En Cloudera AI Workbench SP3:

1. Cree un proyecto desde este repositorio.
2. Abra **Applications > New Application**.
3. Seleccione `app.py` como Script.
4. Seleccione Nvidia GPU Edition con Python 3.10 o 3.11.
5. Asigne una A100 completa de 80 GB, sin MIG.
6. Asigne al menos 8 vCPU y 64 GiB RAM; se recomiendan 16 vCPU y 128 GiB.
7. Configure almacenamiento suficiente para el virtualenv y la caché del modelo.

El código exige `CDSW_APP_PORT` y enlaza exclusivamente a:

```text
127.0.0.1:$CDSW_APP_PORT
```

No configure host ni puerto manualmente.

## Primera ejecución

El primer arranque crea un virtualenv versionado dentro del proyecto, instala vLLM 0.29.0+cu129 y Transformers 5.15.0, ejecuta `pip check`, valida Torch/CUDA y finalmente inicia el servidor. Puede tardar varios minutos. Reinicios posteriores reutilizan el entorno validado y la caché de Hugging Face.

Se necesita acceso HTTPS a GitHub Releases, el índice PyTorch y PyPI, además de Hugging Face para descargar el modelo. En una red cerrada use `VLLM_WHEEL_URL`, `PYTORCH_INDEX_URL`, `QWEN_MODEL_ID` y `HF_HOME` para apuntar a mirrors y snapshots internos.

## Variables recomendadas

| Variable | Default | Notas |
|---|---|---|
| `QWEN_MODEL_ID` | `Qwen/Qwen3.8-27B-FP8` | Modelo o snapshot local |
| `QWEN_SERVED_MODEL_NAME` | `qwen3.8-27b-fp8` | Nombre usado por la API OpenAI |
| `QWEN_MAX_MODEL_LEN` | `262144` | Contexto nativo máximo |
| `QWEN_MAX_NUM_SEQS` | `1` | Prioriza contexto; el servidor puede encolar peticiones |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | `8192` | Bloque de prefill |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.90` | No superar `0.95` |
| `QWEN_KV_CACHE_DTYPE` | `bfloat16` | Obligatorio con Triton en A100 SM80 |
| `QWEN_ATTENTION_BACKEND` | `TRITON_ATTN` | Backend validado |
| `QWEN_ENFORCE_EAGER` | `true` | Perfil conservador; probar `false` separadamente en SP3 |
| `QWEN_CPU_OFFLOAD_GB` | `0` | Mantener cero porque los pesos caben |
| `QWEN_MAX_IMAGES_PER_PROMPT` | `1` | Vídeo permanece deshabilitado |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Evita JIT dependiente de ninja/NVCC |
| `VLLM_ALLOWED_MEDIA_DOMAINS` | vacío | Allowlist CSV para imágenes remotas |
| `QWEN_API_KEY` | vacío | Si se define, vLLM exige Bearer token |
| `QWEN_REASONING_PARSER` | vacío | Opcional; pruebe `qwen3` antes de producción |
| `QWEN_TOOL_CALL_PARSER` | vacío | Opcional; habilita auto tool choice con el parser indicado |
| `HF_TOKEN` | vacío | Guardar como secreto de solo lectura |
| `HF_HOME` | default HF | Recomendado: volumen persistente |
| `QWEN_FORCE_REINSTALL` | `false` | Fuerza reinstalación y revalidación del venv |

La A100 puede ejecutar los pesos FP8 mediante Marlin, pero su SM80 no admite caché KV FP8 con Triton. SP3 no cambia esa limitación física.

## API y streaming

La URL de la Application actúa como base OpenAI. Compruebe primero:

```text
GET /health
GET /v1/models
```

Ejemplo Python:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://SUBDOMINIO.DOMINIO/v1",
    api_key="VALOR_DE_QWEN_API_KEY",
)

stream = client.chat.completions.create(
    model="qwen3.8-27b-fp8",
    messages=[{"role": "user", "content": "Explica RAG en dos frases."}],
    max_tokens=1024,
    temperature=0.7,
    stream=True,
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)
```

La Application y/o el ingress de Cloudera pueden exigir autenticación adicional a la API key de vLLM. Debe validarse el acceso máquina a máquina, la conservación de `Authorization`, el buffering SSE y los timeouts de la instalación concreta.

Si el ingress de Cloudera utiliza el mismo header `Authorization`, no active `QWEN_API_KEY` hasta comprobar que el Bearer token llega intacto a vLLM. Para reasoning y herramientas, empiece con chat normal y habilite posteriormente `QWEN_REASONING_PARSER=qwen3` y `QWEN_TOOL_CALL_PARSER=qwen3_coder`, cada uno en una prueba separada.

## Límites

- Un único proceso vLLM: no use múltiples workers HTTP porque cada uno intentaría cargar otra copia del modelo.
- Una A100 80 GB completa por Application.
- Salida textual; no genera imágenes ni vídeos.
- Una imagen por prompt y cero vídeos con los defaults.
- No aparece automáticamente en el catálogo de Cloudera AI Inference ni en RAG Studio.
- Workbench Applications no sustituyen el autoscaling y ciclo de vida de AI Inference.
- El servidor OpenAI controla `max_tokens` por petición; esta Application no añade el antiguo techo global de 8192 del wrapper `predict`.

## English summary

This project runs Qwen3.8-27B-FP8 as a long-lived Cloudera AI Workbench SP3 Application with vLLM's OpenAI-compatible API and SSE streaming. `app.py` performs the complete bootstrap: it creates an isolated versioned virtual environment, installs and validates the cu129 serving stack, then replaces itself with the vLLM server bound strictly to `127.0.0.1:$CDSW_APP_PORT`.

Use one full A100 80 GB, keep BF16 KV cache on SM80, keep the FlashInfer sampler disabled unless a complete JIT toolchain is validated, and never launch multiple HTTP workers. Test Cloudera ingress authentication, SSE buffering, and connection timeouts before production or RAG Studio integration.
