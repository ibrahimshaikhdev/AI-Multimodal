from __future__ import annotations

import ast
import html
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_KB_DIRECTORY = PROJECT_ROOT / "project_kb"
GENERATED_FILENAMES = (
    "project_overview.md",
    "features.md",
    "api_endpoints.md",
    "data_model.md",
    "how_to_use.md",
)
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "route"}


def _read_source(root: Path, relative_path: str) -> str:
    path = root / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read KB source {relative_path}: {exc}") from exc


def _markdown_section(text: str, title: str) -> str:
    match = re.search(
        rf"^(?P<level>#{{1,6}})\s+{re.escape(title)}\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        return ""
    level = len(match.group("level"))
    next_heading = re.search(
        rf"^#{{1,{level}}}\s+",
        text[match.end() :],
        re.MULTILINE,
    )
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[match.end() : end].strip()


def _overview(readme: str, srs: str) -> str:
    purpose = _markdown_section(readme, "Project purpose")
    workspace = _markdown_section(readme, "Workspace libraries")
    stack = _markdown_section(readme, "Tech stack")
    building = _markdown_section(srs, "1. What We Are Building")
    goals = _markdown_section(srs, "2. Main Goals")
    return (
        "# Project overview\n\n"
        "This is the Multimodal AI Research & Medical Report Intelligence "
        "platform, a Flask web application with a browser-based HTML and "
        "JavaScript interface.\n\n"
        f"{building}\n\n"
        f"{purpose}\n\n"
        f"{workspace}\n\n"
        "## Main capabilities\n\n"
        f"{goals}\n\n"
        "## Technology stack\n\n"
        f"{stack}\n\n"
        "## Safety and privacy\n\n"
        "The platform is a research and experimental information-management "
        "tool, not a clinical diagnosis system. Patient records are associated "
        "with user-owned patient accounts and must be accessed through the "
        "authenticated, ownership-checked application workflows.\n"
    )


def _features(readme: str, srs: str, frontend: str) -> str:
    requirement_rows = [
        line.strip()
        for line in _markdown_section(srs, "3. Official SRS Functional Requirements").splitlines()
        if re.match(r"^\|\s*FR-", line.strip())
    ]
    workspace_section = _markdown_section(readme, "Workspace libraries")
    nav_labels = []
    for match in re.finditer(
        r"<button[^>]*data-workspace-nav=\"([^\"]+)\"[^>]*>([\s\S]*?)</button>",
        frontend,
        re.IGNORECASE,
    ):
        label = re.sub(r"<[^>]+>", " ", match.group(2))
        label = re.sub(r"\s+", " ", html.unescape(label)).strip()
        if label:
            nav_labels.append(f"- **{label}** (`{match.group(1)}`)")

    readme_topics = []
    for match in re.finditer(r"^##\s+(.+)$", readme, re.MULTILINE):
        heading = match.group(1).strip()
        if heading not in {
            "Project purpose",
            "Workspace libraries",
            "Tech stack",
            "Local setup",
            "Environment variables",
            "Cloud text generation",
            "Testing",
            "Team workflow",
            "GitHub handoff",
            "License",
        }:
            readme_topics.append(f"- {heading}")

    return (
        "# Application features\n\n"
        "The signed-in workspace is divided into these frontend sections:\n\n"
        + ("\n".join(nav_labels) if nav_labels else "- Workspace sections are defined in `frontend/index.html`.")
        + "\n\n## Documented capabilities\n\n"
        + f"{workspace_section}\n\n"
        + "\n".join(readme_topics)
        + "\n\n## SRS functional requirements\n\n"
        + ("\n".join(requirement_rows) if requirement_rows else "See `PROJECT_OVERVIEW.md`.")
        + "\n\nMedical image-model outputs are experimental and must not be presented "
        "as confirmed diagnoses.\n"
    )


def _route_endpoint(decorator: ast.expr) -> tuple[str, str] | None:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return None
    method = decorator.func.attr.lower()
    if method not in HTTP_METHODS or not decorator.args:
        return None
    route_path = decorator.args[0]
    if not isinstance(route_path, ast.Constant) or not isinstance(route_path.value, str):
        return None
    if method == "route":
        methods = {"GET"}
        for keyword in decorator.keywords:
            if keyword.arg == "methods" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                methods = {
                    element.value.upper()
                    for element in keyword.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                }
        method_name = ", ".join(sorted(methods))
    else:
        method_name = method.upper()
    path = route_path.value
    if not path.startswith("/api/"):
        path = f"/api{path if path.startswith('/') else '/' + path}"
    return method_name, path


def _api_endpoints(root: Path) -> str:
    endpoints = []
    routes_directory = root / "backend" / "routes"
    for path in sorted(routes_directory.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            raise RuntimeError(f"Could not inspect route source {path.name}: {exc}") from exc
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                endpoint = _route_endpoint(decorator)
                if endpoint:
                    method, route_path = endpoint
                    endpoints.append((path.stem, method, route_path, node.name))

    lines = [
        "# API endpoints",
        "",
        "These routes are discovered from Flask route decorators in `backend/routes/`. "
        "All listed API routes are registered under `/api`.",
        "",
    ]
    previous_module = None
    for module_name, method, route_path, function_name in sorted(endpoints):
        if module_name != previous_module:
            lines.extend((f"## {module_name.replace('_', ' ').title()}", ""))
            previous_module = module_name
        lines.append(f"- `{method} {route_path}` — `{function_name}`")
    return "\n".join(lines) + "\n"


def _data_model(root: Path) -> str:
    model_lines = [
        "# Data model",
        "",
        "Database models are defined in `backend/models/`. These records are "
        "persisted by SQLAlchemy; patient information is scoped through the "
        "patient owner.",
        "",
    ]
    models_directory = root / "backend" / "models"
    for path in sorted(models_directory.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            raise RuntimeError(f"Could not inspect model source {path.name}: {exc}") from exc
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            table_name = None
            columns = []
            for statement in node.body:
                if not isinstance(statement, ast.Assign) or not statement.targets:
                    continue
                target = statement.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__tablename__" and isinstance(statement.value, ast.Constant):
                    table_name = statement.value.value
                value = statement.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "Column"
                ):
                    column_type = (
                        ast.unparse(value.args[0])
                        if value.args
                        else "unspecified type"
                    )
                    columns.append(f"`{target.id}` ({column_type})")
            if table_name or columns:
                model_lines.extend(
                    (
                        f"## {node.name}",
                        "",
                        f"Database table: `{table_name or 'not explicitly named'}`.",
                        "",
                        "Stored fields: "
                        + (", ".join(columns) if columns else "No columns discovered.")
                        + ".",
                        "",
                    )
                )
    return "\n".join(model_lines).rstrip() + "\n"


def _how_to_use(readme: str) -> str:
    setup = _markdown_section(readme, "Local setup")
    return (
        "# How to use the application\n\n"
        "Start the Flask backend from the repository root and open the local "
        "application in a browser. Sign in, then use the workspace navigation "
        "to manage patients, upload reports or scans, and open research tools.\n\n"
        f"{setup}\n\n"
        "## Assistant context\n\n"
        "The own-model Assistant retrieves project information from the Markdown "
        "and text files in `project_kb/`. If the user selects a report or patient "
        "timeline, the Assistant retrieves that signed-in user's selected report "
        "text or dated patient records from the database and includes them as "
        "PATIENT DATA, even when the question does not name the patient. If report "
        "text has not been extracted, it runs the existing document text "
        "extraction service on the uploaded file before answering. Patient "
        "records are retrieved live and are not copied into this shared knowledge "
        "base.\n"
    )


def build_project_kb(root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    root = Path(root)
    readme = _read_source(root, "README.md")
    srs = _read_source(root, "PROJECT_OVERVIEW.md")
    frontend = _read_source(root, "frontend/index.html")
    contents = (
        _overview(readme, srs),
        _features(readme, srs, frontend),
        _api_endpoints(root),
        _data_model(root),
        _how_to_use(readme),
    )
    kb_directory = root / "project_kb"
    try:
        kb_directory.mkdir(parents=True, exist_ok=True)
        (kb_directory / "notes.md").touch(exist_ok=True)
        output_paths = []
        for filename, content in zip(GENERATED_FILENAMES, contents):
            output_path = kb_directory / filename
            output_path.write_text(content, encoding="utf-8")
            output_paths.append(output_path)
    except OSError as exc:
        raise RuntimeError(f"Could not write project knowledge base: {exc}") from exc
    return tuple(output_paths)


if __name__ == "__main__":
    for generated_path in build_project_kb():
        print(generated_path.relative_to(PROJECT_ROOT))
