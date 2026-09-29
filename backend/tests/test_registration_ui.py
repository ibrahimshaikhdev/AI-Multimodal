from pathlib import Path


def test_registration_form_exists_in_frontend():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "Create account" in html
    assert "first_name" in html
    assert "last_name" in html
    assert "email" in html
    assert "password" in html
    assert "/api/auth/register" in html


def test_authentication_views_and_memory_only_token_handling_exist():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'id="loginForm"' in html
    assert 'id="registerForm"' in html
    assert 'id="signedInView"' in html
    assert 'id="logoutButton"' in html
    assert "/api/auth/login" in html
    assert "/api/auth/logout" in html
    assert "let accessToken" in html
    assert "localStorage" not in html
    assert "sessionStorage" not in html


def test_patient_management_views_are_connected_to_the_api():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert 'id="patientsView"' in html
    assert 'id="patientList"' in html
    assert 'id="patientDetail"' in html
    assert 'id="patientForm"' in html
    assert 'id="newPatientButton"' in html
    assert 'id="editPatientButton"' in html
    assert "/api/patients" in html
    assert 'isEditing ? "PUT" : "POST"' in html
    assert "Authorization: `Bearer ${accessToken}`" in html
