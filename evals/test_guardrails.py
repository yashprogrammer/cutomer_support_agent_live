from __future__ import annotations

import pytest

from customer_support_agent.core.settings import Settings
from customer_support_agent.observability import NoOpTracer
from customer_support_agent.services.guardrails_service import GuardrailsService


@pytest.fixture(scope="module")
def service() -> GuardrailsService:
    return GuardrailsService(settings=Settings(guardrails_enabled=True), tracer=NoOpTracer())


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("My debit card number is 4111 1111 1111 1111 and the ATM transaction failed.", "<CARD_NUMBER>"),
        ("Please check account number 123456789012 for this savings account fee issue.", "<ACCOUNT_NUMBER>"),
        ("My bank email update request should go to alex@example.com.", "<EMAIL_ADDRESS>"),
        ("Call me on 9876543210 about the blocked debit card request.", "<PHONE_NUMBER>"),
    ],
)
def test_input_redacts_structured_pii_without_blocking(service: GuardrailsService, text: str, token: str) -> None:
    result = service.check_input(text)
    assert result.passed is True
    assert token in result.sanitized_text
    assert result.to_dict()["pii_redacted"] is True


def test_in_scope_banking_request_passes_without_llm(monkeypatch: pytest.MonkeyPatch, service: GuardrailsService) -> None:
    monkeypatch.setattr(
        service,
        "_classify_scope_with_llm",
        lambda text: (_ for _ in ()).throw(AssertionError("LLM fallback should not run")),
    )
    result = service.check_input("My ATM withdrawal failed and I need help with the reversal timeline.")
    assert result.passed is True


def test_off_topic_request_is_rejected_without_llm(monkeypatch: pytest.MonkeyPatch, service: GuardrailsService) -> None:
    monkeypatch.setattr(
        service,
        "_classify_scope_with_llm",
        lambda text: (_ for _ in ()).throw(AssertionError("LLM fallback should not run")),
    )
    result = service.check_input("Write me a funny poem about a mountain trip.")
    assert result.passed is False
    assert any(v["type"] == "scope_violation" for v in result.violations)


def test_ambiguous_scope_can_pass_with_llm_allow(monkeypatch: pytest.MonkeyPatch, service: GuardrailsService) -> None:
    monkeypatch.setattr(service, "_classify_scope_with_llm", lambda text: "in_scope")
    result = service.check_input("Can you help me with this service request?")
    assert result.passed is True


def test_ambiguous_scope_fails_closed_when_llm_uncertain(monkeypatch: pytest.MonkeyPatch, service: GuardrailsService) -> None:
    monkeypatch.setattr(service, "_classify_scope_with_llm", lambda text: "uncertain")
    result = service.check_input("Can you help me with this service request?")
    assert result.passed is False
    assert any(v["label"] == "uncertain" for v in result.violations if v["type"] == "scope_violation")


def test_safe_output_passes_unchanged(service: GuardrailsService) -> None:
    draft = "Hi Aarti, the ATM reversal is typically processed within 24 hours. Please share the ATM location if it does not reverse by then."
    result = service.check_output(draft)
    assert result.passed is True
    assert result.sanitized_text == draft
    assert result.violations == []


def test_output_redacts_pii_without_blocking(service: GuardrailsService) -> None:
    draft = "Hi, I can see the card ending 4111 1111 1111 1111 on the account."
    result = service.check_output(draft)
    assert result.passed is True
    assert "<CARD_NUMBER>" in result.sanitized_text
    assert result.to_dict()["pii_redacted"] is True


def test_toxic_output_is_blocked(service: GuardrailsService) -> None:
    result = service.check_output("You are an idiot and this is your fault.")
    assert result.passed is False
    assert any(v["type"] == "toxicity_violation" for v in result.violations)


@pytest.mark.parametrize(
    "draft",
    [
        "This investment is 100% safe for you.",
        "You are getting a guaranteed return from this account.",
        "Think of it as risk-free free money.",
    ],
)
def test_forbidden_financial_promises_are_blocked(service: GuardrailsService, draft: str) -> None:
    result = service.check_output(draft)
    assert result.passed is False
    assert any(v["type"] == "promise_violation" for v in result.violations)


def test_guardrails_can_be_disabled() -> None:
    service = GuardrailsService(settings=Settings(guardrails_enabled=False), tracer=NoOpTracer())
    result = service.check_input("Write me a poem and my card is 4111 1111 1111 1111.")
    assert result.passed is True
    assert result.sanitized_text.endswith("4111 1111 1111 1111.")
