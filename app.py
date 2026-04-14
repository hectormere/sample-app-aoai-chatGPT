import os
import json
import uuid
import time
import asyncio
import logging

from quart import Blueprint, Quart, jsonify, make_response, request, send_from_directory, render_template

from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AioDefaultAzureCredential

from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageInputTextBlock

from backend.settings import app_settings
from backend.utils import (
    format_as_ndjson,
    format_non_streaming_response,
)

# ------------------------------------------------------------
# Teams Tabs hardening headers (kept for compatibility)
# ------------------------------------------------------------
FRAME_ANCESTORS = (
    "frame-ancestors "
    "https://teams.microsoft.com "
    "https://*.teams.microsoft.com "
    "https://*.cloud.microsoft;"
)
X_FRAME_OPTIONS = "ALLOW-FROM https://teams.microsoft.com/"

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

# ------------------------------------------------------------
# Foundry Project + Agent resolution by name/version
# ------------------------------------------------------------
AZURE_AI_PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("AZURE_AI_PROJECT_ENDPOINT_STRING")

AGENT_NAME = os.getenv("AZURE_FOUNDRY_AGENT_NAME")
AGENT_VERSION = os.getenv("AZURE_FOUNDRY_AGENT_VERSION")

if not AZURE_AI_PROJECT_ENDPOINT:
    raise RuntimeError("Falta AZURE_AI_PROJECT_ENDPOINT (o AZURE_AI_PROJECT_ENDPOINT_STRING) en variables de entorno.")
if not AGENT_NAME:
    raise RuntimeError("Falta AZURE_FOUNDRY_AGENT_NAME en variables de entorno.")
if not AGENT_VERSION:
    raise RuntimeError("Falta AZURE_FOUNDRY_AGENT_VERSION en variables de entorno.")

_foundry_cred = DefaultAzureCredential(exclude_visual_studio_code_credential=True)
_project_client = AIProjectClient(endpoint=AZURE_AI_PROJECT_ENDPOINT, credential=_foundry_cred)

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
    """Best-effort extractor compatible with current SDK shapes."""
    try:
        data = list_messages_result.as_dict()
    except Exception:
        try:
            data = dict(list_messages_result)
        except Exception:
            data = None

    if not data:
        return ""

    items = data.get("data") or data.get("messages") or data.get("items") or []
    if not isinstance(items, list):
        return ""

    # Sort by timestamp if available
    items = sorted([m for m in items if isinstance(m, dict)], key=_best_ts)

    for m in reversed(items):
        if m.get("role") != "assistant":
            continue

        content = m.get("content")
        # content can be a string or list of blocks
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block and isinstance(block["text"], str):
                        parts.append(block["text"])
                    elif block.get("type") == "text":
                        txt = block.get("text")
                        if isinstance(txt, dict):
                            val = txt.get("value")
                            if isinstance(val, str):
                                parts.append(val)
                        elif isinstance(txt, str):
                            parts.append(txt)
                else:
                    # SDK objects: try common fields
                    t = getattr(block, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
                    elif hasattr(t, "value") and isinstance(getattr(t, "value"), str):
                        parts.append(getattr(t, "value"))
            return "\n".join([p for p in parts if p])

    return ""


def _get_agent_id_by_name_version(name: str, version: str) -> str:
    """Resolve existing agent id using name + version."""
    try:
        agents = _project_client.agents.list_agents()
    except Exception as ex:
        raise RuntimeError(f"No puedo listar agentes del proyecto. {ex}")

    # list_agents may return pageable; iterate safely
    for a in agents:
        ad = a.as_dict() if hasattr(a, "as_dict") else {}
        a_name = getattr(a, "name", None) or ad.get("name")
        a_version = getattr(a, "version", None) or ad.get("version")
        if a_name == name and str(a_version) == str(version):
            a_id = getattr(a, "id", None) or ad.get("id")
            if a_id:
                return a_id

    raise RuntimeError(f"No encuentro el agente name={name} version={version}")


def _run_agent_sync(conversation_id: str, user_text: str) -> str:
    thread_id = _ensure_thread(conversation_id)
    agent_id = _get_agent_id_by_name_version(AGENT_NAME, AGENT_VERSION)

    content = [MessageInputTextBlock(text=user_text or "")]
    _project_client.agents.create_message(thread_id=thread_id, role="user", content=content)

    _project_client.agents.create_and_process_run(thread_id=thread_id, agent_id=agent_id)

    msgs = _project_client.agents.list_messages(thread_id=thread_id)
    return _extract_last_assistant_text(msgs)


# ------------------------------------------------------------
# Minimal response adapter (keeps the sample frontend working)
# ------------------------------------------------------------
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
    history_metadata = request_body.get("history_metadata") or {}
    if isinstance(history_metadata, dict) and "conversation_id" not in history_metadata:
        history_metadata["conversation_id"] = _get_conversation_id(request_body)
    return format_non_streaming_response(response, history_metadata, apim_request_id=None)


async def conversation_internal(request_body, request_headers):
    try:
        # Mantengo tu misma lógica: si stream está activo, el frontend espera NDJSON.
        if getattr(app_settings.azure_openai, "stream", False) and not getattr(app_settings.base_settings, "use_promptflow", False):
            # Streaming real no lo hacemos aquí. Devolvemos 1 chunk NDJSON para compatibilidad.
            response, _ = await send_chat_request(request_body, request_headers)
            history_metadata = request_body.get("history_metadata") or {}
            if isinstance(history_metadata, dict) and "conversation_id" not in history_metadata:
                history_metadata["conversation_id"] = _get_conversation_id(request_body)
            ndjson = format_as_ndjson((_FakeChunk(response.choices[0].message.content, finish_reason="stop"),))
            resp = await make_response(ndjson)
            resp.timeout = None
            resp.mimetype = "application/json-lines"
            return resp
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)
    except Exception as ex:
        logging.exception(ex)
        return jsonify({"error": str(ex)}), 500


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @bp.after_app_request
    def _security_headers(resp):
        resp.headers["Content-Security-Policy"] = FRAME_ANCESTORS
        resp.headers["X-Frame-Options"] = X_FRAME_OPTIONS
        return resp

    @bp.route("/")
    async def index():
        return await render_template(
            "index.html",
            title=app_settings.ui.title,
            favicon=app_settings.ui.favicon,
        )

    @bp.route("/favicon.ico")
    async def favicon():
        return await bp.send_static_file("favicon.ico")

    @bp.route("/assets/<path:path>")
    async def assets(path):
        return await send_from_directory("static/assets", path)

    @bp.route("/frontend_settings", methods=["GET"])
    def frontend_settings():
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

    @bp.route("/conversation", methods=["POST"])
    async def conversation():
        if not request.is_json:
            return jsonify({"error": "request must be json"}), 415
        request_json = await request.get_json()
        return await conversation_internal(request_json, request.headers)

    # Mantengo el resto de endpoints /history/* del sample sin cambios en tu repo.
    return app


app = create_app()
