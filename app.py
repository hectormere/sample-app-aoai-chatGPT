# app.py
# Quart UI original + conversacion con Azure AI Foundry Agent por ID

import json
import os
import logging
import uuid
import asyncio
import time

from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
)

from azure.identity.aio import DefaultAzureCredential as AioDefaultAzureCredential
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageInputTextBlock

from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import app_settings
from backend.utils import (
    format_as_ndjson,
    format_stream_response,
    format_non_streaming_response,
)

# Seguridad Teams Tabs
FRAME_ANCESTORS = (
    "frame-ancestors "
    "https://teams.microsoft.com "
    "https://*.teams.microsoft.com "
    "https://*.cloud.microsoft;"
)
X_FRAME_OPTIONS = "ALLOW-FROM https://teams.microsoft.com/"

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")
cosmos_db_ready = asyncio.Event()

# Foundry Agent
CONN_STR = os.getenv("CONN_STR")
AGENT_ID = os.getenv("AZURE_AGENT_ID") or os.getenv("AZURE_VOICELIVE_AGENT_ID")

if not CONN_STR:
    raise RuntimeError("Falta CONN_STR en variables de entorno.")
if not AGENT_ID:
    raise RuntimeError("Falta AZURE_AGENT_ID (o AZURE_VOICELIVE_AGENT_ID) en variables de entorno.")

_foundry_cred = DefaultAzureCredential(exclude_visual_studio_code_credential=True)
#_project_client = AIProjectClient.from_connection_string(credential=_foundry_cred, conn_str=CONN_STR)
_project_client = AIProjectClient(endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),credential=_foundry_cred)
_openai_client = _project_client.get_openai_client()

_threads = {}  # conversation_id -> thread_id


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


def _run_agent_sync(conversation_id: str, user_text: str) -> str:
    #thread_id = _ensure_thread(conversation_id)
    #agent = _project_client.agents.get_agent(AGENT_ID)
    #content = [MessageInputTextBlock(text=user_text or "")]
    #_project_client.agents.create_message(thread_id=thread_id, role="user", content=content)
    #_project_client.agents.create_and_process_run(thread_id=thread_id, agent_id=agent.id)
    #msgs = _project_client.agents.list_messages(thread_id=thread_id)
    #return _extract_last_assistant_text(msgs)

    resp = _openai_client.responses.create(
        input=[
            {
                "role": "user",
                "content": user_text or ""
            }
        ],
        extra_body={
            "agent_reference": {
                "name": os.getenv("AZURE_FOUNDRY_AGENT_NAME"),
                "version": os.getenv("AZURE_FOUNDRY_AGENT_VERSION"),
                "type": "agent_reference"
            }
        }
    )

    # extracción robusta del texto
    try:
        if getattr(resp, "output_text", None):
            return resp.output_text
    except Exception:
        pass

    try:
        out = resp.to_dict().get("output", [])
        for item in out:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return c.get("text", "")
    except Exception:
        pass

    return ""



def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.after_request
    async def add_security_headers(response):
        response.headers["Content-Security-Policy"] = FRAME_ANCESTORS
        response.headers["X-Frame-Options"] = X_FRAME_OPTIONS
        return response

    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            cosmos_db_ready.set()

    return app


@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


@bp.route("/frontend_settings", methods=["GET"])
def frontend_settings():
    # Fuerza a desactivar historial para que el frontend no llame a /history/* si no está configurado
    try:
        ui = {
            "title": app_settings.ui.title,
            "logo": app_settings.ui.logo,
            "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
            "chat_title": app_settings.ui.chat_title,
            "chat_description": app_settings.ui.chat_description,
            "show_share_button": app_settings.ui.show_share_button,
            "show_chat_history_button": False,
        }
        payload = {
            "auth_enabled": app_settings.base_settings.auth_enabled,
            "feedback_enabled": False,
            "ui": ui,
            "sanitize_answer": app_settings.base_settings.sanitize_answer,
            "oyd_enabled": bool(app_settings.base_settings.datasource_type),
        }
        return jsonify(payload), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


async def init_cosmosdb_client():
    # Mantiene compatibilidad, pero si no está configurado, devuelve None
    if not app_settings.chat_history:
        return None
    try:
        cosmos_endpoint = f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
        if not app_settings.chat_history.account_key:
            async with AioDefaultAzureCredential() as cred:
                credential = cred
        else:
            credential = app_settings.chat_history.account_key

        return CosmosConversationClient(
            cosmosdb_endpoint=cosmos_endpoint,
            credential=credential,
            database_name=app_settings.chat_history.database,
            container_name=app_settings.chat_history.conversations_container,
            enable_message_feedback=app_settings.chat_history.enable_feedback,
        )
    except Exception as e:
        logging.exception("Exception in CosmosDB initialization")
        return None


# Objetos fake para que backend/utils.py pueda formatear NDJSON
class _FakeDelta:
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content


class _FakeChoiceDelta:
    def __init__(self, content: str, finish_reason=None):
        self.index = 0
        self.delta = _FakeDelta(content)
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content: str, finish_reason=None):
        self.id = str(uuid.uuid4())
        self.model = "foundry-agent"
        self.created = int(time.time())
        self.choices = [_FakeChoiceDelta(content, finish_reason=finish_reason)]


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _extract_last_user_text(request_body: dict) -> str:
    msgs = request_body.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _get_conversation_id(request_body: dict) -> str:
    hm = request_body.get("history_metadata") or {}
    if isinstance(hm, dict) and hm.get("conversation_id"):
        return hm.get("conversation_id")
    if request_body.get("conversation_id"):
        return request_body.get("conversation_id")
    return str(uuid.uuid4())


async def send_chat_request(request_body, request_headers):
    conversation_id = _get_conversation_id(request_body)
    user_text = _extract_last_user_text(request_body)
    answer = await asyncio.to_thread(_run_agent_sync, conversation_id, user_text)
    return _FakeResponse(answer), None


async def complete_chat_request(request_body, request_headers):
    response, _ = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {}) or {}
    if isinstance(history_metadata, dict) and "conversation_id" not in history_metadata:
        history_metadata["conversation_id"] = _get_conversation_id(request_body)
    return format_non_streaming_response(response, history_metadata, apim_request_id=None)


async def stream_chat_request(request_body, request_headers):
    response, _ = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {}) or {}
    if isinstance(history_metadata, dict) and "conversation_id" not in history_metadata:
        history_metadata["conversation_id"] = _get_conversation_id(request_body)

    full_text = response.choices[0].message.content or ""

    async def generate(apim_request_id, history_metadata):
        for token in full_text.split(" "):
            yield format_stream_response(_FakeChunk(token + " "), history_metadata, apim_request_id=None)
            await asyncio.sleep(0.01)
        yield format_stream_response(_FakeChunk("", finish_reason="stop"), history_metadata, apim_request_id=None)

    return generate(apim_request_id=None, history_metadata=history_metadata)


async def conversation_internal(request_body, request_headers):
    try:
        if app_settings.azure_openai.stream and not app_settings.base_settings.use_promptflow:
            result = await stream_chat_request(request_body, request_headers)
            response = await make_response(format_as_ndjson(result))
            response.timeout = None
            response.mimetype = "application/json-lines"
            return response
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)
    except Exception as ex:
        logging.exception(ex)
        return jsonify({"error": str(ex)}), 500


@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    return await conversation_internal(request_json, request.headers)


# Mantengo el resto de endpoints /history/* del sample sin cambios en tu repo.

app = create_app()
