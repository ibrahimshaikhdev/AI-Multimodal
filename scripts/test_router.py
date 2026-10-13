import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.assistant_router import classify
from backend.tests.test_assistant_router import ROUTER_CASES


def main() -> None:
    if len(ROUTER_CASES) < 60:
        raise SystemExit("Router evaluation requires at least 60 labelled questions.")
    fallback_cases = sum(expected == "fallback" for _, expected in ROUTER_CASES)
    if fallback_cases < len(ROUTER_CASES) * 0.5:
        raise SystemExit("At least half of router cases must be fallback questions.")

    misses = []
    for question, expected in ROUTER_CASES:
        actual, score = classify(question)
        if actual != expected:
            misses.append((question, expected, actual, score))

    accuracy = (len(ROUTER_CASES) - len(misses)) / len(ROUTER_CASES)
    print(
        f"Router accuracy: {accuracy:.1%} "
        f"({len(ROUTER_CASES) - len(misses)}/{len(ROUTER_CASES)})"
    )
    for question, expected, actual, score in misses:
        print(
            f"MISS: {question!r}: expected {expected}, got {actual} "
            f"(score={score:.3f})"
        )
    if accuracy < 0.90:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
