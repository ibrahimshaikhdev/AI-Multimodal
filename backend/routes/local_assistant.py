import json

from flask import Blueprint, current_app, g, jsonify, request

from backend.auth import require_auth
from backend.extensions import db
from backend.models import LocalAssistantConversation
from backend.services.local_llm import (
    local_context_size,
    local_generate,
    local_token_count,
)
from .ai import (
    AI_QUERY_DISCLAIMER,
    LOCAL_ASSISTANT_MAX_HISTORY_MESSAGES,
    LOCAL_ASSISTANT_SYSTEM_PROMPT,
    _local_assistant_sources,
)


local_assistant_bp = Blueprint("local_assistant", __name__)
LOCAL_ASSISTANT_MAX_OUTPUT_TOKENS = 500
LOCAL_ASSISTANT_CONTEXT_RESERVE_TOKENS = 120


@local_assistant_bp.get("/ai/local-conversation")
@require_auth
def get_local_assistant_conversation():
    conversation = (
        db.session.query(LocalAssistantConversation)
        .filter_by(user_id=g.current_user.id)
        .first()
    )
    if conversation is None:
        return jsonify({"success": True, "messages": []}), 200

    try:
        messages = json.loads(conversation.messages_json)
    except (TypeError, json.JSONDecodeError):
        current_app.logger.exception(
            "Stored local assistant conversation is unreadable for user %s",
            g.current_user.id,
        )
        return jsonify(
            {
                "success": False,
                "error": "Stored local assistant conversation could not be read.",
            }
        ), 500
    if not isinstance(messages, list) or any(
        not isinstance(message, dict)
        or message.get("role") not in {"user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        current_app.logger.error(
            "Stored local assistant conversation has an invalid format for user %s",
            g.current_user.id,
        )
        return jsonify(
            {
                "success": False,
                "error": "Stored local assistant conversation has an invalid format.",
            }
        ), 500
    return jsonify({"success": True, "messages": messages}), 200


@local_assistant_bp.delete("/ai/local-conversation")
@require_auth
def clear_local_assistant_conversation():
    conversation = (
        db.session.query(LocalAssistantConversation)
        .filter_by(user_id=g.current_user.id)
        .first()
    )
    if conversation is not None:
        db.session.delete(conversation)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Could not clear local assistant conversation for user %s",
                g.current_user.id,
            )
            return jsonify(
                {
                    "success": False,
                    "error": "The saved local conversation could not be cleared.",
                }
            ), 500
    return jsonify({"success": True, "messages": []}), 200


def _shrink_local_assistant_context(context):
    sources = context.get("sources", [])
    if len(sources) > 1:
        sources.pop()
        return True
    for source in sources:
        for key in ("extracted_text", "excerpt"):
            value = source.get(key)
            if isinstance(value, str) and value:
                source[key] = value[: max(0, len(value) * 3 // 4)]
                return True
    return False


@local_assistant_bp.post("/ai/query-local")
@require_auth
def query_local_assistant():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON object is required."}), 400

    question = data.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        return jsonify(
            {
                "success": False,
                "error": "question must be a non-empty string of at most 2000 characters.",
            }
        ), 400
    question = question.strip()
    context, sources, context_error = _local_assistant_sources(
        g.current_user.id,
        {**data, "question": question},
    )
    if context_error:
        return context_error

    conversation = (
        db.session.query(LocalAssistantConversation)
        .filter_by(user_id=g.current_user.id)
        .first()
    )
    if conversation is None:
        previous_messages = []
    else:
        try:
            previous_messages = json.loads(conversation.messages_json)
        except (TypeError, json.JSONDecodeError):
            current_app.logger.exception(
                "Stored local assistant conversation is unreadable for user %s",
                g.current_user.id,
            )
            return jsonify(
                {
                    "success": False,
                    "error": "Stored local assistant conversation could not be read.",
                }
            ), 500
        if not isinstance(previous_messages, list) or any(
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
            for message in previous_messages
        ):
            current_app.logger.error(
                "Stored local assistant conversation has an invalid format for user %s",
                g.current_user.id,
            )
            return jsonify(
                {
                    "success": False,
                    "error": "Stored local assistant conversation has an invalid format.",
                }
            ), 500

    history = previous_messages[-LOCAL_ASSISTANT_MAX_HISTORY_MESSAGES:]
    try:
        context_size = local_context_size()
    except Exception as exc:
        current_app.logger.exception(
            "Could not read local model context size for user %s",
            g.current_user.id,
        )
        return jsonify({"success": False, "error": str(exc)}), 502
    prompt_budget = (
        context_size
        - LOCAL_ASSISTANT_MAX_OUTPUT_TOKENS
        - LOCAL_ASSISTANT_CONTEXT_RESERVE_TOKENS
    )
    if prompt_budget < 1:
        return jsonify(
            {
                "success": False,
                "error": "The local model server context is too small for a response. Restart it with a larger context size.",
            }
        ), 502

    while True:
        serialized_context = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        prompt = (
            f"Current source context:\n{serialized_context}\n\n"
            "Conversation history (use it only to understand follow-up questions; "
            "current source context is the authority for facts):\n"
            f"{json.dumps(history, ensure_ascii=False, indent=2)}\n\n"
            f"Current question:\n{question}"
        )
        try:
            prompt_tokens = local_token_count(
                f"{LOCAL_ASSISTANT_SYSTEM_PROMPT}\n{prompt}"
            )
        except Exception as exc:
            current_app.logger.exception(
                "Could not tokenize local assistant prompt for user %s",
                g.current_user.id,
            )
            return jsonify({"success": False, "error": str(exc)}), 502
        if prompt_tokens <= prompt_budget:
            break
        if history:
            history = history[2:] if len(history) > 1 else []
        elif not _shrink_local_assistant_context(context):
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "The question and selected context are too large for the local model's "
                        f"{context_size}-token context window, even after shortening the source excerpts."
                    ),
                }
            ), 413

    try:
        answer = local_generate(
            prompt,
            system=LOCAL_ASSISTANT_SYSTEM_PROMPT,
            max_tokens=LOCAL_ASSISTANT_MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:
        current_app.logger.exception(
            "Local assistant generation failed for user %s",
            g.current_user.id,
        )
        return jsonify(
            {
                "success": False,
                "error": str(exc),
            }
        ), 502

    messages = [
        *previous_messages[-(LOCAL_ASSISTANT_MAX_HISTORY_MESSAGES - 2):],
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    if conversation is None:
        conversation = LocalAssistantConversation(
            user_id=g.current_user.id,
            messages_json=json.dumps(messages, ensure_ascii=False),
        )
        db.session.add(conversation)
    else:
        conversation.messages_json = json.dumps(messages, ensure_ascii=False)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception(
            "Could not save local assistant conversation for user %s",
            g.current_user.id,
        )
        return jsonify(
            {
                "success": False,
                "error": "The response was generated, but the conversation could not be saved.",
            }
        ), 500

    return jsonify(
        {
            "success": True,
            "answer": answer,
            "context_type": context["context_type"],
            "sources": sources,
            "messages": messages,
            "disclaimer": AI_QUERY_DISCLAIMER,
        }
    ), 200
