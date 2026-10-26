import pytest
from pydantic import ValidationError

from app.providers.base import ProviderResponse
from app.services.response_formatter import format_response


def test_format_response_defaults_source_to_document() -> None:
    response = format_response("r1", 10.0, ProviderResponse(text="hi", model="m"))
    assert response.source == "document"


def test_format_response_accepts_web_source() -> None:
    response = format_response("r1", 10.0, ProviderResponse(text="hi", model="m"), source="web")
    assert response.source == "web"


def test_format_response_rejects_invalid_source() -> None:
    with pytest.raises(ValidationError):
        format_response("r1", 10.0, ProviderResponse(text="hi", model="m"), source="bogus")  # type: ignore[arg-type]
