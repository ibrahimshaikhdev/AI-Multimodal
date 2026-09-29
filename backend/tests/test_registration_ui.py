from pathlib import Path


def test_registration_form_exists_in_frontend():
    html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "Register" in html
    assert "first_name" in html
    assert "last_name" in html
    assert "email" in html
    assert "password" in html
    assert "/api/auth/register" in html
