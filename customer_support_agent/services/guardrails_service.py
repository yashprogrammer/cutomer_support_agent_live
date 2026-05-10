from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from guardrails.validators import (
    FailResult,
    PassResult,
    Validator,
    register_validator,
)

from customer_support_agent.core.settings import Settings
from customer_support_agent.observability.tracer import NoOpTracer, Tracer

try:
    from guardrails.hub import DetectPII as _HubDetectPII
except Exception:  # pragma: no cover
    _HubDetectPII = None

try:
    from guardrails.hub import ToxicLanguage as _HubToxicLanguage
except Exception:  # pragma: no cover
    _HubToxicLanguage = None

try:
    from guardrails.hub import RestrictToTopic as _HubRestrictToTopic
except Exception:  # pragma: no cover
    _HubRestrictToTopic = None


_PII_REPLACEMENTS = {
    "CARD_NUMBER": "<CARD_NUMBER>",
    "ACCOUNT_NUMBER": "<ACCOUNT_NUMBER>",
    "EMAIL_ADDRESS": "<EMAIL_ADDRESS>",
    "PHONE_NUMBER": "<PHONE_NUMBER>",
}

# guardrails-ai DetectPII emits Presidio entity tokens (e.g. <CREDIT_CARD>),
# but the rest of the system uses our own canonical token names. This map
# rewrites DetectPII output and infers entity_types for the violation dict.
_HUB_TOKEN_ALIASES = {
    "<CREDIT_CARD>": ("<CARD_NUMBER>", "CARD_NUMBER"),
    "<CARD_NUMBER>": ("<CARD_NUMBER>", "CARD_NUMBER"),
    "<EMAIL_ADDRESS>": ("<EMAIL_ADDRESS>", "EMAIL_ADDRESS"),
    "<PHONE_NUMBER>": ("<PHONE_NUMBER>", "PHONE_NUMBER"),
    "<US_PHONE_NUMBER>": ("<PHONE_NUMBER>", "PHONE_NUMBER"),
    "<ACCOUNT_NUMBER>": ("<ACCOUNT_NUMBER>", "ACCOUNT_NUMBER"),
}


def _normalize_pii_tokens(text: str) -> tuple[str, list[str]]:
    """Rewrite hub-style PII tokens to our canonical tokens and report entities."""
    normalized = text
    seen: list[str] = []
    for hub_token, (canonical_token, entity) in _HUB_TOKEN_ALIASES.items():
        if hub_token in normalized:
            seen.append(entity)
            if hub_token != canonical_token:
                normalized = normalized.replace(hub_token, canonical_token)
    return normalized, seen


@register_validator(name="csa/account-number-redact", data_type="string")
class AccountNumberValidator(Validator):
    PATTERN = re.compile(
        r"(?:(?:account|a/c)(?: number| no\.?)?[:\s-]*)\b\d{8,18}\b",
        flags=re.IGNORECASE,
    )
    DIGIT_RUN = re.compile(r"\d{8,18}")

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        text = str(value or "")
        matches = list(self.PATTERN.finditer(text))
        if not matches:
            return PassResult()
        fixed = self.PATTERN.sub(
            lambda m: self.DIGIT_RUN.sub(_PII_REPLACEMENTS["ACCOUNT_NUMBER"], m.group(0)),
            text,
        )
        return FailResult(
            error_message="Detected bank account number(s).",
            fix_value=fixed,
            metadata={"entity_types": ["ACCOUNT_NUMBER"], "count": len(matches)},
        )


@register_validator(name="csa/regex-pii-fallback", data_type="string")
class RegexPiiValidator(Validator):
    PATTERNS: dict[str, re.Pattern[str]] = {
        "CARD_NUMBER": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "EMAIL_ADDRESS": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            flags=re.IGNORECASE,
        ),
        "PHONE_NUMBER": re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){10,12})"),
    }

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        text = str(value or "")
        sanitized = text
        entities: list[str] = []
        total = 0
        for entity_type, pattern in self.PATTERNS.items():
            found = list(pattern.finditer(sanitized))
            if not found:
                continue
            entities.append(entity_type)
            total += len(found)
            sanitized = pattern.sub(_PII_REPLACEMENTS[entity_type], sanitized)
        if not entities:
            return PassResult()
        return FailResult(
            error_message="Detected structured PII via regex.",
            fix_value=sanitized,
            metadata={"entity_types": entities, "count": total},
        )


@register_validator(name="csa/toxic-language-regex", data_type="string")
class ToxicLanguageRegexValidator(Validator):
    PATTERNS: list[re.Pattern[str]] = [
        re.compile(p, flags=re.IGNORECASE)
        for p in (
            r"\bidiot\b",
            r"\bmoron\b",
            r"\bstupid\b",
            r"\bshut up\b",
            r"\bdamn you\b",
            r"\bhell with you\b",
            r"\bfool\b",
        )
    ]

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        text = str(value or "")
        matches: list[str] = []
        for pattern in self.PATTERNS:
            matches.extend(m.group(0) for m in pattern.finditer(text))
        if not matches:
            return PassResult()
        return FailResult(
            error_message="Draft contains hostile or abusive language.",
            metadata={"matches": sorted(set(matches))},
        )


@register_validator(name="csa/forbidden-promises", data_type="string")
class ForbiddenPhrasesValidator(Validator):
    PATTERNS: list[re.Pattern[str]] = [
        re.compile(p, flags=re.IGNORECASE)
        for p in (
            r"\bguaranteed return\b",
            r"\bguaranteed profit\b",
            r"\bfree money\b",
            r"\b100%\s+safe\b",
            r"\brisk[- ]free\b",
            r"\bzero[- ]risk\b",
            r"\bcan(?:not|'t)? lose\b",
            r"\bdouble your money\b",
        )
    ]

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        text = str(value or "")
        matches: list[str] = []
        for pattern in self.PATTERNS:
            matches.extend(m.group(0) for m in pattern.finditer(text))
        if not matches:
            return PassResult()
        return FailResult(
            error_message="Draft makes forbidden financial guarantees or promises.",
            metadata={"matches": sorted(set(matches))},
        )


@dataclass
class GuardrailResult:
    passed: bool
    sanitized_text: str
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pii_redacted"] = any(
            violation.get("type") == "pii_redaction" for violation in self.violations
        )
        return payload


class GuardrailsService:
    ESCALATION_MESSAGE = (
        "Thanks for reaching out. A support specialist needs to review this request before "
        "we send a draft response."
    )

    _BANKING_KEYWORDS = {
        "account",
        "atm",
        "balance",
        "bank",
        "banking",
        "billing",
        "branch",
        "card",
        "cash",
        "charge",
        "charges",
        "customer",
        "debit",
        "deposit",
        "fee",
        "fees",
        "ifsc",
        "interest",
        "kyc",
        "loan",
        "minimum balance",
        "net banking",
        "otp",
        "passbook",
        "payment",
        "pin",
        "plan",
        "priority queue",
        "refund",
        "savings",
        "sla",
        "support",
        "ticket",
        "transaction",
        "update my email",
        "update my mobile",
        "withdraw",
        "withdrawal",
    }
    _OFF_TOPIC_KEYWORDS = {
        "blog post",
        "code",
        "haiku",
        "joke",
        "movie",
        "poem",
        "recipe",
        "song",
        "story",
        "travel",
        "weather",
        "write me",
    }

    def __init__(self, settings: Settings, tracer: TracerLike | None = None):
        self._settings = settings
        self._enabled = settings.guardrails_enabled
        self._tracer = tracer or NoOpTracer()
        self._classifier_llm: ChatGroq | None = None

        self._pii_validators: list[Validator] = []
        self._toxicity_validator: Validator | None = None
        self._forbidden_validator: Validator | None = None
        self._scope_validator: Validator | None = None
        self._setup_validators()

    def _setup_validators(self) -> None:
        # Account-number validator runs FIRST so context-aware pattern
        # ("account number 123...") captures digits before generic PII regexes
        # can mis-classify them as phone numbers.
        self._pii_validators.append(AccountNumberValidator(on_fail="fix"))

        if _HubDetectPII is not None:
            try:
                self._pii_validators.append(
                    _HubDetectPII(
                        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
                        on_fail="fix",
                    )
                )
            except Exception:
                pass
        if len(self._pii_validators) == 1:
            self._pii_validators.append(RegexPiiValidator(on_fail="fix"))

        if _HubToxicLanguage is not None:
            try:
                self._toxicity_validator = _HubToxicLanguage(
                    threshold=0.5, validation_method="sentence", on_fail="noop"
                )
            except Exception:
                self._toxicity_validator = None
        if self._toxicity_validator is None:
            self._toxicity_validator = ToxicLanguageRegexValidator(on_fail="noop")

        self._forbidden_validator = ForbiddenPhrasesValidator(on_fail="noop")

        if _HubRestrictToTopic is not None:
            try:
                self._scope_validator = _HubRestrictToTopic(
                    valid_topics=[
                        "banking",
                        "account servicing",
                        "atm",
                        "card",
                        "kyc",
                        "fees and charges",
                        "support ticket",
                    ],
                    invalid_topics=[
                        "poetry",
                        "creative writing",
                        "weather",
                        "recipes",
                        "travel",
                        "song",
                        "code",
                    ],
                    disable_classifier=False,
                    disable_llm=True,
                    on_fail="noop",
                )
            except Exception:
                self._scope_validator = None

    def check_input(self, text: str) -> GuardrailResult:
        if not self._enabled:
            return GuardrailResult(passed=True, sanitized_text=str(text or ""), violations=[])

        sanitized_text, pii_violations = self.sanitize_text(text)
        classification = self._classify_scope(text=sanitized_text)

        violations = list(pii_violations)
        if classification["label"] != "in_scope":
            violations.append(
                {
                    "type": "scope_violation",
                    "reason": classification["reason"],
                    "label": classification["label"],
                }
            )
            return GuardrailResult(
                passed=False,
                sanitized_text=sanitized_text,
                violations=violations,
            )

        return GuardrailResult(
            passed=True,
            sanitized_text=sanitized_text,
            violations=violations,
        )

    def check_output(self, text: str) -> GuardrailResult:
        if not self._enabled:
            return GuardrailResult(passed=True, sanitized_text=str(text or ""), violations=[])

        sanitized_text, pii_violations = self.sanitize_text(text)
        violations = list(pii_violations)

        toxic_result = self._toxicity_validator.validate(sanitized_text, {}) if self._toxicity_validator else PassResult()
        if isinstance(toxic_result, FailResult):
            metadata = toxic_result.metadata or {}
            matches = metadata.get("matches") or [toxic_result.error_message]
            violations.append(
                {
                    "type": "toxicity_violation",
                    "matches": list(matches),
                    "reason": "Draft contains hostile or abusive language.",
                }
            )

        promise_result = self._forbidden_validator.validate(sanitized_text, {}) if self._forbidden_validator else PassResult()
        if isinstance(promise_result, FailResult):
            metadata = promise_result.metadata or {}
            matches = metadata.get("matches") or []
            violations.append(
                {
                    "type": "promise_violation",
                    "matches": list(matches),
                    "reason": "Draft makes forbidden financial guarantees or promises.",
                }
            )

        blocked = any(
            violation.get("type") in {"toxicity_violation", "promise_violation"}
            for violation in violations
        )

        return GuardrailResult(
            passed=not blocked,
            sanitized_text=sanitized_text,
            violations=violations,
        )

    def sanitize_text(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        source = str(text or "")
        if not self._enabled:
            return source, []

        sanitized_text = source
        all_entities: list[str] = []
        total_count = 0
        for validator in self._pii_validators:
            result = validator.validate(sanitized_text, {})
            if isinstance(result, FailResult):
                if result.fix_value is not None:
                    sanitized_text = str(result.fix_value)
                metadata = result.metadata or {}
                entity_types = list(metadata.get("entity_types") or [])
                count = int(metadata.get("count") or 0)
                if not entity_types:
                    sanitized_text, inferred = _normalize_pii_tokens(sanitized_text)
                    entity_types = inferred
                    count = len(inferred)
                else:
                    sanitized_text, _ = _normalize_pii_tokens(sanitized_text)
                all_entities.extend(entity_types)
                total_count += count or len(entity_types)

        violations: list[dict[str, Any]] = []
        if all_entities:
            violations.append(
                {
                    "type": "pii_redaction",
                    "entity_types": sorted(set(all_entities)),
                    "count": total_count,
                }
            )
        return sanitized_text, violations

    def _classify_scope(self, text: str) -> dict[str, str]:
        lowered = f" {text.lower()} "
        has_scope_keyword = any(keyword in lowered for keyword in self._BANKING_KEYWORDS)
        has_off_topic_keyword = any(keyword in lowered for keyword in self._OFF_TOPIC_KEYWORDS)

        if has_scope_keyword and not has_off_topic_keyword:
            return {"label": "in_scope", "reason": "deterministic keyword match"}
        if has_off_topic_keyword and not has_scope_keyword:
            return {"label": "off_topic", "reason": "deterministic off-topic keyword match"}

        llm_label = self._classify_scope_with_llm(text=text)
        if llm_label == "in_scope":
            return {"label": "in_scope", "reason": "llm fallback classifier allowed request"}
        if llm_label == "off_topic":
            return {"label": "off_topic", "reason": "llm fallback classifier rejected request"}

        return {"label": "uncertain", "reason": "scope classifier could not determine request scope"}

    def _classify_scope_with_llm(self, text: str) -> str:
        if self._scope_validator is not None:
            try:
                result = self._scope_validator.validate(text, {})
                if isinstance(result, FailResult):
                    return "off_topic"
                if isinstance(result, PassResult):
                    return "in_scope"
            except Exception:
                pass

        if not self._settings.groq_api_key:
            return "uncertain"

        classifier = self._get_classifier_llm()
        if classifier is None:
            return "uncertain"

        system_prompt = (
            "You classify customer-support requests. Reply with exactly one label: "
            "IN_SCOPE, OFF_TOPIC, or UNCERTAIN.\n"
            "IN_SCOPE means the request is about banking support, account servicing, banking "
            "policies, card/ATM issues, KYC/profile updates, billing/plan checks, or support "
            "ticket/account triage.\n"
            "OFF_TOPIC means it asks for unrelated creative writing, entertainment, recipes, "
            "travel, weather, coding help, or general chat.\n"
            "Use UNCERTAIN only when the text is too ambiguous to classify."
        )

        with self._tracer.start_span("scope_classifier_invoke", redacted_input=True) as span:
            span["prompt"] = {
                "system": system_prompt,
                "user": text,
            }
            response = classifier.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=text),
                ]
            )
            label = str(getattr(response, "content", response)).strip().upper()
            span["response"] = label

        if "IN_SCOPE" in label:
            return "in_scope"
        if "OFF_TOPIC" in label:
            return "off_topic"
        return "uncertain"

    def _get_classifier_llm(self) -> ChatGroq | None:
        if self._classifier_llm is not None:
            return self._classifier_llm

        try:
            self._classifier_llm = ChatGroq(
                model=self._settings.groq_model,
                groq_api_key=self._settings.groq_api_key,
                temperature=0.0,
            )
        except Exception:
            self._classifier_llm = None
        return self._classifier_llm


TracerLike = Tracer | NoOpTracer
