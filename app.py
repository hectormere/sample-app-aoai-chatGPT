# app.py
import os
import io
import json
import asyncio
import base64
import logging
import getpass
from typing import Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from PIL import Image

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    MessageInputTextBlock,
    MessageInputImageUrlBlock,
    MessageImageUrlParam,
)

# -----------------------------
# Configuración y logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

RETRIEVAL_EXTS = {
    ".c", ".cpp", ".cs", ".css", ".doc", ".docx", ".go", ".html", ".java", ".js",
    ".json", ".md", ".pdf", ".php", ".pptx", ".py", ".rb", ".sh", ".tex", ".ts", ".txt"
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# Variables de entorno requeridas
CONN_STR = os.getenv("CONN_STR")
AGENT_ID = os.getenv("AZURE_AGENT_ID") or os.getenv("AZURE_VOICELIVE_AGENT_ID")

if not CONN_STR:
    raise RuntimeError("Falta CONN_STR (cadena de conexión del AI Project).")
if not AGENT_ID:
    raise RuntimeError("Falta AZURE_AGENT_ID (o AZURE_VOICELIVE_AGENT_ID).")

# Cliente Foundry (SDK síncrono)
_credential = DefaultAzureCredential(exclude_visual_studio_code_credential=True)
_project_client = AIProjectClient.from_connection_string(credential=_credential, conn_str=CONN_STR)
_agent = _project_client.agents.get_agent(AGENT_ID)

# Estado en memoria por conversación
_threads: Dict[str, str] = {}                 # conversation_id -> thread_id
_pending_images: Dict[str, List[dict]] = {}   # conversation_id -> [{"data_url": "...", "bytes": N, "mime": "..."}]
_uploaded_files: Dict[str, List[dict]] = {}   # opcional: solo meta de lo subido

# -----------------------------
# Utilidades
# -----------------------------
def _ensure_thread(conversation_id: str) -> str:
    if conversation_id not in _threads:
        th = _project_client.agents.create_thread()
        _threads[conversation_id] = th.id
    return _threads[conversation_id]

def _best_ts(m: dict):
    v = m.get("created_at")
    if isinstance(v, (int, float)):
        return v
    v = m.get("createdAt")
    if isinstance(v, (int, float)):
        return v
    return -1

def _extract_last_assistant_text(list_messages_result) -> str:
    """Extrae texto del último mensaje del asistente desde list_messages()."""
    try:
        md = list_messages_result.as_dict()
    except Exception:
        md = None

    if isinstance(md, dict) and isinstance(md.get("data"), list):
        data = [m for m in md["data"] if isinstance(m, dict)]
        assistant_msgs = [m for m in data if m.get("role") == "assistant"]
        if not assistant_msgs:
            return ""
        assistant_msgs.sort(key=_best_ts)
        chosen = assistant_msgs[-1]
        content = chosen.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = (part.get("text") or {}).get("value")
                    if txt:
                        texts.append(txt)
            if texts:
                return "\n".join(texts)
        t = chosen.get("text")
        if isinstance(t, dict) and t.get("value"):
            return t["value"]
    return ""

def _image_to_data_url(raw: bytes) -> dict:
    """Convierte imagen a JPEG comprimido y devuelve data_url + meta."""
    try:
        im = Image.open(io.BytesIO(raw))
    except Exception:
        b64 = base64.b64encode(raw).decode("utf-8")
        return {"mime": "application/octet-stream",
                "data_url": f"data:application/octet-stream;base64,{b64}",
                "bytes": len(raw)}
    max_w = 1024
    if im.width > max_w:
        new_h = int(im.height * (max_w / im.width))
        im = im.resize((max_w, new_h))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=75)
    data = out.getvalue()
    b64 = base64.b64encode(data).decode("utf-8")
    return {"mime": "image/jpeg",
            "data_url": f"data:image/jpeg;base64,{b64}",
            "bytes": len(data)}

def _run_agent_sync(conversation_id: str, user_text: str) -> str:
    """Ejecución síncrona del agente; se invoca desde un hilo con asyncio.to_thread."""
    thread_id = _ensure_thread(conversation_id)
    blocks = [MessageInputTextBlock(text=user_text or "")]
    for img in _pending_images.get(conversation_id, []):
        blocks.append(MessageInputImageUrlBlock(image_url=MessageImageUrlParam(url=img["data_url"])))
    _pending_images[conversation_id] = []

    # Crea mensaje y run
    content_to_send = blocks if len(blocks) > 1 else user_text
    _project_client.agents.create_message(thread_id=thread_id, role="user", content=content_to_send)
    _project_client.agents.create_and_process_run(thread_id=thread_id, agent_id=_agent.id)
    msgs = _project_client.agents.list_messages(thread_id=thread_id)
    return _extract_last_assistant_text(msgs)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

# Static
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.isdir("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

# CSP para que Teams pueda embeber (iframe)
class TeamsIframeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "frame-ancestors https://teams.microsoft.com https://*.teams.microsoft.com https://*.office.com;"
        )
        response.headers["X-Frame-Options"] = "ALLOW-FROM https://teams.microsoft.com/"
        return response

app.add_middleware(TeamsIframeMiddleware)

# -----------------------------
# Rutas
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    # Sirve index.html de la raíz si existe; si no, un HTML mínimo
    if os.path.isfile("index.html"):
        return FileResponse("index.html", media_type="text/html")
    html = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"/><title>Agente</title></head>
<body><h1>Agente de Soporte</h1><p>Sube index.html a la raíz para tu UI.</p></body></html>"""
    return HTMLResponse(html)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "agent_id": AGENT_ID}

@app.get("/tools/userinfo")
def tool_userinfo():
    # En App Service no hay 'az' CLI; devolvemos info básica
    upn = os.getenv("WEBSITE_OWNER_NAME")  # best-effort
    return {"display_name": getpass.getuser(), "upn": upn, "mail": upn}

@app.post("/api/upload")
async def upload_files(conversation_id: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se recibieron ficheros")
    uploaded, skipped_for_retrieval, accepted_images = [], [], []
    try:
        for up in files:
            suffix = os.path.splitext(up.filename)[-1].lower() or ""
            raw = await up.read()
            meta = {
                "filename": up.filename,
                "ext": suffix,
                "size_bytes": len(raw),
                "retrieval_supported": suffix in RETRIEVAL_EXTS,
                "is_image": suffix in IMAGE_EXTS,
            }
            uploaded.append(meta)
            if suffix in IMAGE_EXTS:
                data = _image_to_data_url(raw)
                item = {
                    "filename": up.filename,
                    "mime": data["mime"],
                    "bytes": data["bytes"],
                    "data_url": data["data_url"],
                }
                _pending_images.setdefault(conversation_id, []).append(item)
                accepted_images.append({"filename": up.filename, "mime": data["mime"], "bytes": data["bytes"]})
            if suffix not in RETRIEVAL_EXTS:
                skipped_for_retrieval.append(meta)
        _uploaded_files.setdefault(conversation_id, []).extend(uploaded)
        return {
            "uploaded": uploaded,
            "accepted_images": accepted_images,
            "skipped_for_retrieval": skipped_for_retrieval,
        }
    except Exception as e:
        logger.exception("Error subiendo ficheros")
        raise HTTPException(status_code=500, detail=f"Error subiendo ficheros: {e}")

@app.get("/api/chat/stream")
async def chat_stream(conversation_id: str, message: str):
    # Ejecuta el agente y “streamea” el texto troceado para tu UI
    try:
        text = await asyncio.to_thread(_run_agent_sync, conversation_id, message)
    except Exception as e:
        logger.exception("Error ejecutando el agente")
        return JSONResponse(status_code=500, content={"error": str(e)})

    async def event_gen():
        # troceo simple por espacios para simular SSE incremental
        for token in (text or "").split(" "):
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")