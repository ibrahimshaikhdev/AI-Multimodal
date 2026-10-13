import json
import os
import re
from difflib import SequenceMatcher

from flask import Blueprint, current_app, g, jsonify, request

from backend.auth import require_auth
from backend.extensions import db
from backend.models import LocalAssistantConversation
from backend.services.assistant_desks import (
    MISSING_SELECTION,
    _prepare_patient_data,
    format_timeline_answer,
    generate_grounded_answer,
    handle_fixed_intent,
    retrieve_project_knowledge,
)
from backend.services.assistant_router import (
    classify,
    log_unmatched_question,
    normalize,
)
from backend.services.web_search import search_web
from .ai import (
    AI_QUERY_DISCLAIMER,
    LOCAL_ASSISTANT_MAX_HISTORY_MESSAGES,
    _local_assistant_sources,
)


local_assistant_bp = Blueprint("local_assistant", __name__)
_MEDICAL_RECOMMENDATION_PATTERN = re.compile(
    r"\b(?:recommend\w*|should|must|need to|advis\w*|suggest\w*|"
    r"dietary|repeat testing|follow[- ]?up\s+(?:testing|tests|appointment|"
    r"care|visit|with|in)|consult)\b",
    re.IGNORECASE,
)


def _repeats_recent_answer(answer, previous_messages):
    normalized_answer = re.sub(r"\W+", " ", answer.casefold()).strip()
    if not normalized_answer:
        return False
    recent_answers = [
        message["content"]
        for message in reversed(previous_messages)
        if message["role"] == "assistant"
    ][:3]
    return any(
        SequenceMatcher(
            None,
            normalized_answer,
            re.sub(r"\W+", " ", previous.casefold()).strip(),
        ).ratio()
        >= 0.78
        for previous in recent_answers
    )


def _has_unsupported_recommendation(answer, patient_data):
    normalized_source = re.sub(r"\W+", " ", patient_data.casefold()).strip()
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if not _MEDICAL_RECOMMENDATION_PATTERN.search(sentence):
            continue
        normalized_sentence = re.sub(r"\W+", " ", sentence.casefold()).strip()
        if normalized_sentence not in normalized_source:
            return True
    return False


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
    web_search_enabled = data.get("web_search", False)
    if not isinstance(web_search_enabled, bool):
        return jsonify(
            {"success": False, "error": "web_search must be a boolean."}
        ), 400
    context_type = data.get("context_type")
    if context_type not in {"research", "report", "patient_timeline"}:
        return jsonify(
            {
                "success": False,
                "error": "context_type must be research, report, or patient_timeline.",
            }
        ), 400
    route_context = {
        "context_type": context_type,
        "patient_id": data.get("patient_id"),
        "report_id": data.get("report_id"),
    }
    intent, score = classify(question, route_context)
    normalized_question = normalize(question)
    if intent == "fallback":
        try:
            log_unmatched_question(
                question,
                g.current_user.id,
                current_app.instance_path,
            )
        except RuntimeError as exc:
            current_app.logger.exception(
                "Could not record unmatched Assistant question for user %s",
                g.current_user.id,
            )
            return jsonify({"success": False, "error": str(exc)}), 500

    if os.environ.get("ASSISTANT_DEBUG", "").strip() == "1":
        current_app.logger.info(
            "Assistant router: question=%r normalized=%r intent=%s score=%.3f",
            question,
            normalized_question,
            intent,
            score,
        )

    needs_patient = intent in {
        "patient_values",
        "timeline",
        "scans",
        "compare_reports",
    }
    if intent == "selected_report" and not route_context["report_id"]:
        answer = MISSING_SELECTION
        context = {"context_type": context_type}
        sources = []
    elif needs_patient and not route_context["patient_id"]:
        answer = MISSING_SELECTION
        context = {"context_type": context_type}
        sources = []
    else:
        answer = None
        context = {"context_type": context_type}
        sources = []
        if intent in {
            "selected_report",
            "patient_values",
            "timeline",
            "scans",
            "compare_reports",
        }:
            source_context_type = (
                "report" if intent == "selected_report" else "patient_timeline"
            )
            if intent == "timeline":
                report_limit = None
                scan_limit = None
            elif intent == "compare_reports":
                report_limit = None
                scan_limit = 2
            elif intent == "scans":
                report_limit = 0
                scan_limit = 10
            else:
                report_limit = 10
                scan_limit = 2
            context, sources, context_error = _local_assistant_sources(
                g.current_user.id,
                {
                    **data,
                    "question": question,
                    "context_type": source_context_type,
                },
                report_limit=report_limit,
                scan_limit=scan_limit,
            )
            if context_error:
                error_response, error_status = context_error
                error_body = error_response.get_json(silent=True) or {}
                if (
                    intent == "selected_report"
                    and error_status == 422
                    and "no readable text" in str(error_body.get("error", "")).casefold()
                ):
                    answer = "This report has no readable text."
                    context = {"context_type": context_type}
                    sources = []
                elif (
                    intent in {"scans", "timeline"}
                    and error_status == 422
                    and "no reports or scans" in str(error_body.get("error", "")).casefold()
                ):
                    context = {"context_type": "patient_timeline"}
                    sources = []
                    context["user_id"] = g.current_user.id
                    context["patient_id"] = route_context["patient_id"]
                else:
                    return context_error
            else:
                context["user_id"] = g.current_user.id
                context["patient_id"] = route_context["patient_id"]

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

    web_search_results = []
    web_search_error = None
    if web_search_enabled:
        try:
            web_search_results = search_web(question)
        except Exception as exc:
            web_search_error = str(exc)
            current_app.logger.warning(
                "SerpApi web search failed for assistant user %s: %s",
                g.current_user.id,
                web_search_error,
            )

    try:
        if (
            answer is None
            and web_search_enabled
            and web_search_results
            and intent in {"fallback", "research"}
        ):
            answer, final_prompt = generate_grounded_answer(
                intent,
                question,
                g.current_user.id,
                project_knowledge="(none)",
                web_search_results=web_search_results,
                max_tokens=300,
            )
            if os.environ.get("ASSISTANT_DEBUG", "").strip() == "1":
                current_app.logger.info(
                    "Assistant final prompt (intent=%s):\n%s",
                    intent,
                    final_prompt,
                )

        if answer is None:
            answer = handle_fixed_intent(intent, question, g.current_user.id)

        if intent == "project_info" and answer is None:
            project_knowledge = retrieve_project_knowledge(
                question,
                g.current_user.id,
                top_k=2,
                min_score=0.08,
            )
            if not project_knowledge:
                answer = "I don't have that information in the project notes."
            else:
                answer, final_prompt = generate_grounded_answer(
                    intent,
                    question,
                    g.current_user.id,
                    project_knowledge=project_knowledge,
                    max_tokens=200,
                )
                if os.environ.get("ASSISTANT_DEBUG", "").strip() == "1":
                    current_app.logger.info(
                        "Assistant final prompt (intent=%s):\n%s",
                        intent,
                        final_prompt,
                    )
        elif intent in {
            "selected_report",
            "patient_values",
            "timeline",
            "scans",
            "compare_reports",
        } and answer is None:
            if intent == "timeline":
                answer = format_timeline_answer(context)
            else:
                patient_data = _prepare_patient_data(intent, context)
            if intent == "selected_report" and not patient_data:
                answer = "This report has no readable text."
            elif intent != "timeline" and answer is None:
                max_tokens_by_intent = {
                    "selected_report": 350,
                    "timeline": 400,
                    "compare_reports": 350,
                    "scans": 250,
                    "patient_values": 250,
                }
                answer, final_prompt = generate_grounded_answer(
                    intent,
                    question,
                    g.current_user.id,
                    patient_data=patient_data,
                    max_tokens=max_tokens_by_intent[intent],
                )
                if _repeats_recent_answer(answer, previous_messages):
                    previous_answer = next(
                        message["content"]
                        for message in reversed(previous_messages)
                        if message["role"] == "assistant"
                    )
                    answer, final_prompt = generate_grounded_answer(
                        intent,
                        question,
                        g.current_user.id,
                        patient_data=patient_data,
                        max_tokens=max_tokens_by_intent[intent],
                        previous_answer=previous_answer,
                    )
                if _repeats_recent_answer(answer, previous_messages):
                    answer = (
                        "I couldn't produce a distinct selected-record answer "
                        "without repeating the earlier response. Please ask "
                        "about a specific result or phrase in the selected record."
                    )
                if _has_unsupported_recommendation(answer, patient_data):
                    answer = (
                        "The selected record does not substantiate that "
                        "recommendation. I can help explain specific results "
                        "or wording shown in it."
                    )
                if os.environ.get("ASSISTANT_DEBUG", "").strip() == "1":
                    current_app.logger.info(
                        "Assistant final prompt (intent=%s):\n%s",
                        intent,
                        final_prompt,
                    )
        if answer is None:
            answer = (
                "I can help with patient reports, lab values, timelines, scans, "
                "research papers, medical terms, and questions about this project."
            )
    except Exception as exc:
        current_app.logger.exception(
            "Own-model assistant generation failed for user %s",
            g.current_user.id,
        )
        return jsonify(
            {
                "success": False,
                "error": str(exc),
            }
        ), 502

    assistant_message = {"role": "assistant", "content": answer}
    if web_search_enabled:
        assistant_message["web_search_results"] = web_search_results
        assistant_message["web_search_error"] = web_search_error
    messages = [
        *previous_messages[-(LOCAL_ASSISTANT_MAX_HISTORY_MESSAGES - 2):],
        {"role": "user", "content": question},
        assistant_message,
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
            "intent": intent,
            "context_type": context["context_type"],
            "sources": sources,
            "web_search_enabled": web_search_enabled,
            "web_search_results": web_search_results,
            "web_search_error": web_search_error,
            "messages": messages,
            "disclaimer": AI_QUERY_DISCLAIMER,
        }
    ), 200
