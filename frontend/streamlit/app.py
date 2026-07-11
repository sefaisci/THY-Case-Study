"""THY-branded three-area Streamlit interface for the Agentic RAG API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from api_client import ApiClient, ApiError
from components import document_summary, render_citations, render_usage_details
from theme import THY_CSS

FRONTEND_DIR = Path(__file__).resolve().parent
LOGO_PATH = FRONTEND_DIR / "thy-logo.png"
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")


def load_display_logo(path: Path) -> Image.Image:
    """Remove transparent canvas padding without changing the supplied logo."""
    with Image.open(path) as source:
        logo = source.convert("RGBA")
    alpha_bounds = logo.getchannel("A").getbbox()
    return logo.crop(alpha_bounds) if alpha_bounds else logo


LOGO_IMAGE = load_display_logo(LOGO_PATH)

st.set_page_config(
    page_title="THY Document Intelligence",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(THY_CSS, unsafe_allow_html=True)


@st.cache_resource
def api_client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


client = api_client(BACKEND_BASE_URL)


def initialize_state() -> None:
    defaults = {
        "loaded_username": None,
        "documents": [],
        "chats": [],
        "active_chat_id": None,
        "messages": [],
        "active_jobs": {},
        "job_notifications": set(),
        "model_catalog": None,
        "last_turn_usage": {},
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_api_error(exc: ApiError) -> None:
    st.error(str(exc))


def refresh_workspace(*, keep_active_chat: bool = True) -> None:
    username = st.session_state.loaded_username
    if not username:
        return
    st.session_state.documents = client.list_documents(username)
    st.session_state.chats = client.list_chats(username)
    chat_ids = {item["id"] for item in st.session_state.chats}
    if not keep_active_chat or st.session_state.active_chat_id not in chat_ids:
        st.session_state.active_chat_id = (
            st.session_state.chats[0]["id"] if st.session_state.chats else None
        )
    load_active_messages()


def load_active_messages() -> None:
    username = st.session_state.loaded_username
    chat_id = st.session_state.active_chat_id
    if not username or not chat_id:
        st.session_state.messages = []
        return
    messages = client.list_messages(username, chat_id)
    st.session_state.messages = messages
    for message in messages:
        if (
            message["role"] != "assistant"
            or message["id"] in st.session_state.last_turn_usage
        ):
            continue
        usage = client.get_usage(
            username,
            session_id=chat_id,
            message_id=message["id"],
        )
        st.session_state.last_turn_usage[message["id"]] = {
            "request": usage.get("request"),
            "session": usage.get("session"),
            "total": usage.get("total"),
        }


def load_username(username: str) -> None:
    resolved = client.resolve_user(username)
    st.session_state.loaded_username = resolved["username"]
    st.session_state.active_chat_id = None
    st.session_state.messages = []
    st.session_state.active_jobs = {}
    st.session_state.job_notifications = set()
    st.session_state.last_turn_usage = {}
    st.session_state.uploader_key += 1
    refresh_workspace(keep_active_chat=False)


def ensure_model_catalog() -> dict[str, Any]:
    if st.session_state.model_catalog is None:
        st.session_state.model_catalog = client.get_models()
    return st.session_state.model_catalog


def poll_ingestion_jobs() -> None:
    username = st.session_state.loaded_username
    if not username or not st.session_state.active_jobs:
        return
    terminal = False
    active_jobs = dict(st.session_state.active_jobs)
    jobs = client.get_ingestion_jobs(username, list(active_jobs))
    for job in jobs:
        job_id = job["id"]
        filename = active_jobs[job_id]
        if job["status"] == "completed":
            if job_id not in st.session_state.job_notifications:
                st.success("Ingestion completed successfully.")
                st.caption(
                    f"Ingestion usage: {int(job.get('total_tokens', 0)):,} tokens · "
                    f"known cost ${float(job.get('cost_usd', 0.0)):.6f}"
                )
                st.session_state.job_notifications.add(job_id)
            terminal = True
            st.session_state.active_jobs.pop(job_id, None)
        elif job["status"] == "failed":
            if job_id not in st.session_state.job_notifications:
                st.error(f"Ingestion failed for {filename}: {job.get('failure_message') or 'Unknown error'}")
                st.session_state.job_notifications.add(job_id)
            terminal = True
            st.session_state.active_jobs.pop(job_id, None)
    if terminal:
        st.session_state.documents = client.list_documents(username)


initialize_state()

left, center, right = st.columns([0.22, 0.50, 0.28], gap="large")

with left:
    st.markdown('<span class="thy-column-marker thy-left-marker"></span>', unsafe_allow_html=True)
    st.image(LOGO_IMAGE, width=64)
    st.markdown(
        '<div class="thy-brand-title">Document Intelligence</div>',
        unsafe_allow_html=True,
    )

    entered_username = st.text_input(
        "Username",
        value=st.session_state.loaded_username or "",
        placeholder="Username",
        label_visibility="collapsed",
        key="username_entry",
    ).strip()
    if entered_username and entered_username.casefold() != (
        st.session_state.loaded_username or ""
    ).casefold():
        try:
            load_username(entered_username)
            st.rerun()
        except ApiError as exc:
            show_api_error(exc)
    elif not entered_username and st.session_state.loaded_username:
        st.session_state.loaded_username = None
        st.session_state.documents = []
        st.session_state.chats = []
        st.session_state.active_chat_id = None
        st.session_state.messages = []
        st.session_state.active_jobs = {}
        st.session_state.job_notifications = set()
        st.session_state.last_turn_usage = {}
        st.session_state.uploader_key += 1
        st.rerun()

    if st.button(
        "+  New Chat",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.loaded_username,
    ):
        try:
            chat = client.create_chat(st.session_state.loaded_username)
            st.session_state.active_chat_id = chat["id"]
            refresh_workspace()
            st.rerun()
        except ApiError as exc:
            show_api_error(exc)

    st.markdown("### Chats")
    if not st.session_state.loaded_username:
        st.caption("Enter a username to load chats.")
    elif not st.session_state.chats:
        st.caption("No chat sessions yet.")
    for chat in st.session_state.chats:
        active = chat["id"] == st.session_state.active_chat_id
        if st.button(
            chat["title"],
            key=f"chat-{chat['id']}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_chat_id = chat["id"]
            try:
                load_active_messages()
                st.rerun()
            except ApiError as exc:
                show_api_error(exc)

with right:
    st.markdown('<span class="thy-column-marker thy-right-marker"></span>', unsafe_allow_html=True)
    st.markdown("## Documents & Ingestion")
    st.markdown('<div class="thy-section-rule"></div>', unsafe_allow_html=True)
    try:
        catalog = ensure_model_catalog()
    except ApiError as exc:
        catalog = {"provider_available": False, "models": [], "error": str(exc)}
    available_models = [item["id"] for item in catalog.get("models", [])]
    model_labels = {
        item["id"]: item.get("display_name") or item["id"]
        for item in catalog.get("models", [])
    }
    efforts_by_model = {
        item["id"]: item.get("reasoning_efforts", [])
        for item in catalog.get("models", [])
    }
    unavailable_models = catalog.get("unavailable_models", [])
    unavailable_gpt_56 = [
        item for item in unavailable_models if item.get("family") == "gpt-5.6"
    ]
    if not catalog.get("provider_available"):
        st.error(catalog.get("error") or "OpenAI model availability is unavailable.")
    elif not available_models:
        st.warning("None of the configured model family is available to this OpenAI account.")
    pending_files = st.file_uploader(
        "Upload source documents",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help="Files remain pending until you click Ingest.",
        key=f"source-uploads-{st.session_state.uploader_key}",
    )
    if pending_files:
        st.caption("Pending upload list")
        for pending in pending_files:
            st.write(f"• {pending.name} — {pending.size / 1024:.1f} KB")

    method_label = st.radio(
        "Ingestion Method",
        ["Semantic Chunking", "Docling Fixed Chunking"],
        horizontal=True,
    )
    ingestion_method = "semantic" if method_label == "Semantic Chunking" else "docling"

    model_options = available_models or ["No available models"]
    semantic_model = None
    semantic_effort = None
    if ingestion_method == "semantic":
        semantic_model = st.selectbox(
            "Semantic Chunking Model",
            model_options,
            format_func=lambda model_id: model_labels.get(model_id, model_id),
            disabled=not available_models,
        )
        semantic_efforts = efforts_by_model.get(semantic_model, [])
        semantic_effort = st.selectbox(
            "Semantic Chunking Reasoning Effort",
            semantic_efforts or ["No supported effort"],
            disabled=not semantic_efforts,
        )

    chat_model = st.selectbox(
        "Chat Model",
        model_options,
        format_func=lambda model_id: model_labels.get(model_id, model_id),
        key="chat_model_selector",
        disabled=not available_models,
    )
    chat_efforts = efforts_by_model.get(chat_model, [])
    chat_effort = st.selectbox(
        "Chat Reasoning Effort",
        chat_efforts or ["No supported effort"],
        key="chat_effort_selector",
        disabled=not chat_efforts,
    )
    if unavailable_gpt_56:
        st.caption(
            "GPT-5.6 preview is configured, but this OpenAI project does not "
            "currently have access."
        )
        with st.expander("GPT-5.6 preview details"):
            for item in unavailable_gpt_56:
                label = item.get("display_name") or item["id"]
                st.markdown(f"**{label}** · `{item['id']}`")
                if item.get("description"):
                    st.caption(item["description"])
                st.caption("Enabled reasoning efforts: low, medium, high")
    if st.button(
        "Refresh Model Access",
        use_container_width=True,
        help="Bypass the backend model cache and check this OpenAI project again.",
    ):
        try:
            st.session_state.model_catalog = client.get_models(refresh=True)
            st.rerun()
        except ApiError as exc:
            show_api_error(exc)
    scope_label = st.selectbox(
        "Collection Scope",
        ["Both Collections", "Semantic Chunks", "Docling Fixed Chunks"],
    )
    collection_scope = {
        "Both Collections": "both",
        "Semantic Chunks": "semantic",
        "Docling Fixed Chunks": "docling",
    }[scope_label]
    st.session_state.chat_model = chat_model
    st.session_state.chat_effort = chat_effort
    st.session_state.collection_scope = collection_scope

    if st.button(
        "Ingest",
        type="primary",
        use_container_width=True,
        disabled=(
            not st.session_state.loaded_username
            or not pending_files
            or (ingestion_method == "semantic" and not available_models)
        ),
    ):
        try:
            upload_response = client.upload_documents(
                st.session_state.loaded_username,
                pending_files,
                ingestion_method=ingestion_method,
                semantic_model=semantic_model,
                semantic_reasoning_effort=semantic_effort,
            )
            document_ids = [item["id"] for item in upload_response["documents"]]
            job_response = client.start_ingestion(
                st.session_state.loaded_username,
                document_ids,
            )
            filename_by_id = {
                item["id"]: item["filename"] for item in upload_response["documents"]
            }
            for job in job_response["jobs"]:
                st.session_state.active_jobs[job["id"]] = filename_by_id[job["document_id"]]
            st.session_state.documents = client.list_documents(st.session_state.loaded_username)
            st.session_state.uploader_key += 1
            st.info("Ingestion started. Use Refresh Status to poll progress.")
            st.rerun()
        except ApiError as exc:
            show_api_error(exc)

    refresh_column, count_column = st.columns([0.48, 0.52], vertical_alignment="center")
    with refresh_column:
        if st.button(
            "Refresh Status",
            use_container_width=True,
            disabled=not st.session_state.loaded_username,
        ):
            try:
                poll_ingestion_jobs()
                st.session_state.documents = client.list_documents(st.session_state.loaded_username)
            except ApiError as exc:
                show_api_error(exc)
    with count_column:
        st.caption(f"{len(st.session_state.documents)} document(s)")

    try:
        poll_ingestion_jobs()
    except ApiError as exc:
        show_api_error(exc)

    if not st.session_state.loaded_username:
        st.caption("Enter a username to load documents.")
    elif not st.session_state.documents:
        st.caption("No uploaded documents yet.")
    for document in st.session_state.documents:
        st.markdown(document_summary(document), unsafe_allow_html=True)
        if document["status"] != "deleted":
            if st.button(
                "Delete",
                key=f"delete-{document['id']}",
                use_container_width=True,
            ):
                try:
                    client.delete_document(st.session_state.loaded_username, document["id"])
                    st.session_state.documents = client.list_documents(st.session_state.loaded_username)
                    st.rerun()
                except ApiError as exc:
                    show_api_error(exc)

with center:
    st.title("Cabin Knowledge Assistant")
    st.markdown('<div class="thy-section-rule"></div>', unsafe_allow_html=True)
    if not st.session_state.loaded_username:
        st.markdown(
            '<div class="thy-empty">Enter a username in the left navigation to load your private document workspace.</div>',
            unsafe_allow_html=True,
        )
    elif not st.session_state.active_chat_id:
        st.markdown(
            '<div class="thy-empty">Create a new chat to start a clean short-term memory context.</div>',
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            avatar = str(LOGO_PATH) if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_citations(message.get("citations", []))
                    usage = st.session_state.last_turn_usage.get(message["id"], {})
                    render_usage_details(
                        usage.get("request"),
                        usage.get("session"),
                        usage.get("total"),
                        model=message.get("model"),
                        reasoning_effort=message.get("reasoning_effort"),
                    )

        with st.form("question-form", clear_on_submit=True):
            question = st.text_area(
                "Ask a question about your documents",
                height=90,
                placeholder="Ask a question about your documents",
                label_visibility="collapsed",
            )
            submit = st.form_submit_button(
                "Send",
                type="primary",
                use_container_width=True,
                disabled=not available_models,
            )
        if submit:
            if not question.strip():
                st.warning("Enter a question before sending.")
            else:
                try:
                    with st.spinner("Retrieving grounded evidence and generating an answer..."):
                        turn = client.send_message(
                            st.session_state.loaded_username,
                            st.session_state.active_chat_id,
                            question=question.strip(),
                            chat_model=st.session_state.chat_model,
                            chat_reasoning_effort=st.session_state.chat_effort,
                            collection_scope=st.session_state.collection_scope,
                        )
                    assistant_id = turn["assistant_message"]["id"]
                    st.session_state.last_turn_usage[assistant_id] = {
                        "request": turn["request_usage"],
                        "session": turn["session_usage"],
                        "total": turn["total_usage"],
                    }
                    refresh_workspace()
                    st.rerun()
                except ApiError as exc:
                    show_api_error(exc)
