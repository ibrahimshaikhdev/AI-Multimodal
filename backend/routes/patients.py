from datetime import date, datetime

from flask import Blueprint, g, jsonify, request

from backend.auth import require_auth
from backend.extensions import db
from backend.models.patient import Patient

patients_bp = Blueprint("patients", __name__)


def _serialize_patient(patient):
    return {
        "id": patient.id,
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "date_of_birth": patient.date_of_birth.isoformat() if patient.date_of_birth else None,
        "created_at": patient.created_at.isoformat(),
        "updated_at": patient.updated_at.isoformat(),
    }


def _validate_patient_data(data):
    if not isinstance(data, dict):
        return None, "A JSON object is required"

    values = {}
    for field in ("first_name", "last_name"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            return None, f"{field} is required"
        value = value.strip()
        if len(value) > 80:
            return None, f"{field} must be 80 characters or fewer"
        values[field] = value

    date_value = data.get("date_of_birth")
    if date_value is None:
        values["date_of_birth"] = None
    elif isinstance(date_value, str):
        try:
            parsed_date = datetime.strptime(date_value, "%Y-%m-%d").date()
        except ValueError:
            return None, "date_of_birth must use YYYY-MM-DD format"
        if parsed_date.isoformat() != date_value:
            return None, "date_of_birth must use YYYY-MM-DD format"
        if parsed_date > date.today():
            return None, "date_of_birth cannot be in the future"
        values["date_of_birth"] = parsed_date
    else:
        return None, "date_of_birth must use YYYY-MM-DD format or be null"

    return values, None


def _owned_patient(patient_id):
    return db.session.query(Patient).filter_by(
        id=patient_id,
        created_by_id=g.current_user.id,
    ).first()


@patients_bp.get("/patients")
@require_auth
def list_patients():
    patients = (
        db.session.query(Patient)
        .filter_by(created_by_id=g.current_user.id)
        .order_by(Patient.created_at.desc(), Patient.id.desc())
        .all()
    )
    return jsonify({"patients": [_serialize_patient(patient) for patient in patients]}), 200


@patients_bp.post("/patients")
@require_auth
def create_patient():
    data, error = _validate_patient_data(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    patient = Patient(**data, created_by=g.current_user)
    db.session.add(patient)
    db.session.commit()

    return jsonify({"patient": _serialize_patient(patient)}), 201


@patients_bp.get("/patients/<int:patient_id>")
@require_auth
def get_patient(patient_id):
    patient = _owned_patient(patient_id)
    if patient is None:
        return jsonify({"error": "Patient not found"}), 404

    return jsonify({"patient": _serialize_patient(patient)}), 200


@patients_bp.put("/patients/<int:patient_id>")
@require_auth
def update_patient(patient_id):
    patient = _owned_patient(patient_id)
    if patient is None:
        return jsonify({"error": "Patient not found"}), 404

    data, error = _validate_patient_data(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    patient.first_name = data["first_name"]
    patient.last_name = data["last_name"]
    patient.date_of_birth = data["date_of_birth"]
    db.session.commit()

    return jsonify({"patient": _serialize_patient(patient)}), 200