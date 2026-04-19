from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from customer_support_agent.core.settings import Settings
from customer_support_agent.observability.tracer import NoOpTracer, Tracer

try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except Exception:  # pragma: no cover - import safety for lightweight environments
    AnalyzerEngine = None
    AnonymizerEngine = None
    NlpEngineProvider = None
    OperatorConfig = None
    Pattern = None
    PatternRecognizer = None


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
    _TOXIC_PATTERNS = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bidiot\b",
            r"\bmoron\b",
            r"\bstupid\b",
            r"\bshut up\b",
            r"\bdamn you\b",
            r"\bhell with you\b",
            r"\bfool\b",
        )
    ]
    _FORBIDDEN_PROMISE_PATTERNS = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
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
    _PII_PATTERNS = {
        "CARD_NUMBER": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "ACCOUNT_NUMBER": re.compile(
            r"(?:(?:account|a/c)(?: number| no\.?)?[:\s-]*)\b\d{8,18}\b",
            flags=re.IGNORECASE,
        ),
        "EMAIL_ADDRESS": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            flags=re.IGNORECASE,
        ),
        "PHONE_NUMBER": re.compile(
            r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){10,12})",
            flags=re.IGNORECASE,
        ),
    }
    _PII_REPLACEMENTS = {
        "CARD_NUMBER": "<CARD_NUMBER>",
        "ACCOUNT_NUMBER": "<ACCOUNT_NUMBER>",
        "EMAIL_ADDRESS": "<EMAIL_ADDRESS>",
        "PHONE_NUMBER": "<PHONE_NUMBER>",
    }

    def __init__(self, settings: Settings, tracer: TracerLike | None = None):
        self._settings = settings
        self._enabled = settings.guardrails_enabled
        self._tracer = tracer or NoOpTracer()
        self._classifier_llm: ChatGroq | None = None
        self._analyzer = None
        self._anonymizer = None
        self._setup_presidio()

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

        toxic_matches = self._find_matches(self._TOXIC_PATTERNS, sanitized_text)
        if toxic_matches:
            violations.append(
                {
                    "type": "toxicity_violation",
                    "matches": toxic_matches,
                    "reason": "Draft contains hostile or abusive language.",
                }
            )

        promise_matches = self._find_matches(self._FORBIDDEN_PROMISE_PATTERNS, sanitized_text)
        if promise_matches:
            violations.append(
                {
                    "type": "promise_violation",
                    "matches": promise_matches,
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
        violations: list[dict[str, Any]] = []

        presidio_entities: list[dict[str, Any]] = []
        if self._analyzer and self._anonymizer and OperatorConfig is not None:
            analyzer_results = self._analyzer.analyze(
                text=source,
                language="en",
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
            )
            if analyzer_results:
                operators = {
                    "EMAIL_ADDRESS": OperatorConfig(
                        "replace", {"new_value": self._PII_REPLACEMENTS["EMAIL_ADDRESS"]}
                    ),
                    "PHONE_NUMBER": OperatorConfig(
                        "replace", {"new_value": self._PII_REPLACEMENTS["PHONE_NUMBER"]}
                    ),
                    "CREDIT_CARD": OperatorConfig(
                        "replace", {"new_value": self._PII_REPLACEMENTS["CARD_NUMBER"]}
                    ),
                }
                anonymized = self._anonymizer.anonymize(
                    text=source,
                    analyzer_results=analyzer_results,
                    operators=operators,
                )
                sanitized_text = anonymized.text
                presidio_entities = [
                    {
                        "entity_type": result.entity_type,
                        "start": result.start,
                        "end": result.end,
                    }
                    for result in analyzer_results
                ]

        regex_entities: list[dict[str, Any]] = []
        for entity_type, pattern in self._PII_PATTERNS.items():
            if entity_type == "ACCOUNT_NUMBER":
                matches = list(pattern.finditer(sanitized_text))
                if matches:
                    regex_entities.extend(
                        {
                            "entity_type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                        }
                        for match in matches
                    )
                    sanitized_text = pattern.sub(
                        lambda match: re.sub(
                            r"\d{8,18}",
                            self._PII_REPLACEMENTS[entity_type],
                            match.group(0),
                        ),
                        sanitized_text,
                    )
                continue

            matches = list(pattern.finditer(sanitized_text))
            if matches:
                regex_entities.extend(
                    {
                        "entity_type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                    }
                    for match in matches
                )
                sanitized_text = pattern.sub(self._PII_REPLACEMENTS[entity_type], sanitized_text)

        all_entities = presidio_entities + regex_entities
        if all_entities:
            grouped_types = sorted({item["entity_type"] for item in all_entities})
            violations.append(
                {
                    "type": "pii_redaction",
                    "entity_types": grouped_types,
                    "count": len(all_entities),
                }
            )

        return sanitized_text, violations

    def _setup_presidio(self) -> None:
        if AnalyzerEngine is None or AnonymizerEngine is None or NlpEngineProvider is None:
            return

        try:
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )
            analyzer = AnalyzerEngine(
                nlp_engine=provider.create_engine(),
                supported_languages=["en"],
            )
            if Pattern is not None and PatternRecognizer is not None:
                analyzer.registry.add_recognizer(
                    PatternRecognizer(
                        supported_entity="ACCOUNT_NUMBER",
                        patterns=[
                            Pattern(
                                name="bank_account_number",
                                regex=r"\b\d{8,18}\b",
                                score=0.45,
                            )
                        ],
                    )
                )
            self._analyzer = analyzer
            self._anonymizer = AnonymizerEngine()
        except Exception:
            self._analyzer = None
            self._anonymizer = None

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

    @staticmethod
    def _find_matches(patterns: list[re.Pattern[str]], text: str) -> list[str]:
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(match.group(0) for match in pattern.finditer(text))
        return sorted(set(matches))


TracerLike = Tracer | NoOpTracer
