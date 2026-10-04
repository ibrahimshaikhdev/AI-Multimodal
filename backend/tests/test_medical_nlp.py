import pytest

from backend.services import (
    AssertionStatus,
    MedicalEntityMention,
    MedicalNLPResult,
    MedicalNLPService,
)


class ExampleProvider:
    def __init__(self, mentions):
        self.mentions = mentions
        self.received_text = None

    def analyze(self, text):
        self.received_text = text
        return self.mentions


def test_medical_nlp_service_returns_common_result_from_injected_provider():
    text = "Possible meniscal tear"
    mention = MedicalEntityMention(
        text="meniscal tear",
        label="condition",
        start=9,
        end=22,
        confidence=0.91,
        assertion=AssertionStatus.POSSIBLE,
    )
    provider = ExampleProvider([mention])

    result = MedicalNLPService(provider).analyze(text)

    assert isinstance(result, MedicalNLPResult)
    assert provider.received_text == text
    assert result.mentions == (mention,)


def test_medical_nlp_service_rejects_blank_text_without_calling_provider():
    provider = ExampleProvider([])

    with pytest.raises(ValueError, match="Text content is required"):
        MedicalNLPService(provider).analyze(" \n\t ")

    assert provider.received_text is None


@pytest.mark.parametrize(
    "mention",
    [
        MedicalEntityMention(text="tear", label="condition", start=5, end=None),
        MedicalEntityMention(text="tear", label="condition", confidence=1.2),
        MedicalEntityMention(text="tear", label="condition", start="0", end=4),
        MedicalEntityMention(text="tear", label="condition", confidence=float("nan")),
    ],
)
def test_medical_nlp_service_rejects_invalid_provider_mentions(mention):
    provider = ExampleProvider([mention])

    with pytest.raises(ValueError):
        MedicalNLPService(provider).analyze("Possible tear")