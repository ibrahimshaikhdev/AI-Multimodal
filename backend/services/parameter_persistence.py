from __future__ import annotations

import hashlib
import json
from datetime import date

from backend.extensions import db
from backend.models.report_parameter import ReportParameter


def _fingerprint(parameter: dict) -> str:
    identity = [
        str(parameter["parameter"]).strip().casefold(),
        str(parameter["value"]).strip().casefold(),
        str(parameter["unit"]).strip().casefold(),
        " ".join(str(parameter["source"]).split()).casefold(),
    ]
    encoded_identity = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded_identity.encode("utf-8")).hexdigest()


def persist_report_parameters(report_id: int, parameters: list[dict]) -> int:
    if not parameters:
        return 0

    fingerprints = {_fingerprint(parameter) for parameter in parameters}
    existing = set(
        fingerprint
        for (fingerprint,) in db.session.query(ReportParameter.fingerprint)
        .filter(
            ReportParameter.report_id == report_id,
            ReportParameter.fingerprint.in_(fingerprints),
        )
        .all()
    )

    inserted = 0
    seen = set(existing)
    for parameter in parameters:
        fingerprint = _fingerprint(parameter)
        if fingerprint in seen:
            continue

        result_date_value = parameter.get("date")
        result_date = date.fromisoformat(result_date_value) if result_date_value else None
        db.session.add(
            ReportParameter(
                report_id=report_id,
                parameter=parameter["parameter"],
                value=parameter["value"],
                unit=parameter["unit"],
                result_date=result_date,
                confidence=parameter.get("confidence"),
                source=parameter["source"],
                fingerprint=fingerprint,
            )
        )
        seen.add(fingerprint)
        inserted += 1

    return inserted


__all__ = ["persist_report_parameters"]