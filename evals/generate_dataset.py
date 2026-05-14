from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customer_support_agent.core.settings import Settings

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
OUTPUT_PATH = DATASET_DIR / "golden.json"

CASE_BLUEPRINTS: list[dict[str, Any]] = [
    {
        "id": "atm_standard_limit",
        "ticket": {
            "subject": "Daily ATM limit for my regular savings account",
            "description": "Please confirm how much cash I can withdraw from an ATM today from a standard savings account.",
            "priority": "medium",
        },
        "customer": {"email": "aarti.shah@example.com", "name": "Aarti Shah", "company": "Northwind"},
        "expected_answer": "Standard savings accounts can withdraw up to INR 25,000 per day at ATMs, though the exact cap can vary by card type and risk profile.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "atm_premium_limit",
        "ticket": {
            "subject": "Premium savings ATM cash cap",
            "description": "I have a premium savings account and want to know the current daily ATM withdrawal limit.",
            "priority": "medium",
        },
        "customer": {"email": "rahul.verma@example.com", "name": "Rahul Verma", "company": "Northwind"},
        "expected_answer": "Premium savings accounts can usually withdraw up to INR 50,000 per day, subject to card and risk-profile limits.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "atm_cash_debited_reversal",
        "ticket": {
            "subject": "ATM debited my account but gave no cash",
            "description": "The ATM deducted money but did not dispense cash. How long does reversal normally take?",
            "priority": "high",
        },
        "customer": {"email": "meera.jain@example.com", "name": "Meera Jain", "company": "Northwind"},
        "expected_answer": "ATM cash-not-dispensed reversals are typically processed within 24 hours.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "atm_cash_debited_dispute",
        "ticket": {
            "subject": "What details are needed for an ATM dispute?",
            "description": "My ATM transaction was debited without cash. If the reversal does not happen in time, what details should I share for a dispute?",
            "priority": "high",
        },
        "customer": {"email": "vivek.nanda@example.com", "name": "Vivek Nanda", "company": "Northwind"},
        "expected_answer": "If the ATM reversal does not happen within 24 hours, the dispute should include the transaction date and time, ATM location, and the last 4 digits of the card.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "atm_wrong_pin_auto_unblock",
        "ticket": {
            "subject": "Card blocked after wrong PIN attempts",
            "description": "My debit card got blocked after I entered the wrong PIN three times. Does it unblock automatically?",
            "priority": "medium",
        },
        "customer": {"email": "neha.bose@example.com", "name": "Neha Bose", "company": "Northwind"},
        "expected_answer": "Cards blocked after three incorrect PIN attempts are typically unblocked automatically after 24 hours.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "atm_wrong_pin_manual_reset",
        "ticket": {
            "subject": "Need help after PIN lock",
            "description": "I entered the wrong ATM PIN too many times. Is there a manual reset option or do I have to wait?",
            "priority": "medium",
        },
        "customer": {"email": "sanjay.kapoor@example.com", "name": "Sanjay Kapoor", "company": "Northwind"},
        "expected_answer": "After three wrong PIN attempts, the card can either auto-unblock after 24 hours or the customer can request a manual reset.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "atm_pin_safety_guidance",
        "ticket": {
            "subject": "Can the bank ask me for my PIN?",
            "description": "Someone claiming to be from the bank asked for my ATM PIN and OTP. Is that expected?",
            "priority": "high",
        },
        "customer": {"email": "pooja.iyer@example.com", "name": "Pooja Iyer", "company": "Northwind"},
        "expected_answer": "Bank staff should never ask for the full card PIN, and customers should never share their OTP or PIN.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "urban_min_balance",
        "ticket": {
            "subject": "Urban branch minimum balance",
            "description": "What monthly average balance is required for an urban branch savings account?",
            "priority": "medium",
        },
        "customer": {"email": "karan.mehra@example.com", "name": "Karan Mehra", "company": "Northwind"},
        "expected_answer": "Urban branch savings accounts require a minimum monthly average balance of INR 10,000.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "semiurban_min_balance",
        "ticket": {
            "subject": "Semi-urban branch balance requirement",
            "description": "Please confirm the minimum monthly average balance for a semi-urban savings account.",
            "priority": "medium",
        },
        "customer": {"email": "lina.patel@example.com", "name": "Lina Patel", "company": "Northwind"},
        "expected_answer": "Semi-urban branch savings accounts require a minimum monthly average balance of INR 5,000.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "rural_min_balance",
        "ticket": {
            "subject": "Rural branch minimum balance",
            "description": "What is the monthly average balance threshold for a rural branch savings account?",
            "priority": "medium",
        },
        "customer": {"email": "arun.dev@example.com", "name": "Arun Dev", "company": "Northwind"},
        "expected_answer": "Rural branch savings accounts require a minimum monthly average balance of INR 2,500.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "non_maintenance_fee",
        "ticket": {
            "subject": "Penalty for not maintaining balance",
            "description": "What fee is usually charged if my savings account average balance falls below the required threshold?",
            "priority": "medium",
        },
        "customer": {"email": "fatima.khan@example.com", "name": "Fatima Khan", "company": "Northwind"},
        "expected_answer": "Typical non-maintenance charges are INR 350 plus applicable taxes and are posted at the month-end billing cycle.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "sms_alert_fee",
        "ticket": {
            "subject": "SMS alert annual fee",
            "description": "Can you confirm the annual SMS alert charge on a savings account?",
            "priority": "low",
        },
        "customer": {"email": "sara.lobo@example.com", "name": "Sara Lobo", "company": "Northwind"},
        "expected_answer": "The annual SMS alert fee is INR 150 plus taxes.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "debit_card_fee",
        "ticket": {
            "subject": "Debit card maintenance charges",
            "description": "What is the yearly debit card maintenance fee for my account?",
            "priority": "low",
        },
        "customer": {"email": "jatin.arora@example.com", "name": "Jatin Arora", "company": "Northwind"},
        "expected_answer": "The debit card annual maintenance fee is INR 250 plus taxes.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "closure_after_14_days",
        "ticket": {
            "subject": "Account closure charges after two weeks",
            "description": "I opened my savings account more than 14 days ago. Is there any account closure fee now?",
            "priority": "low",
        },
        "customer": {"email": "mohit.taneja@example.com", "name": "Mohit Taneja", "company": "Northwind"},
        "expected_answer": "There is no closure fee after 14 days from account opening.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "closure_within_14_days",
        "ticket": {
            "subject": "Closing a newly opened savings account",
            "description": "I want to close a savings account within the first two weeks. What processing charge applies?",
            "priority": "low",
        },
        "customer": {"email": "isha.garg@example.com", "name": "Isha Garg", "company": "Northwind"},
        "expected_answer": "Closing the account within the first 14 days may attract an INR 200 processing charge.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "low_risk_kyc_frequency",
        "ticket": {
            "subject": "KYC refresh cycle for low-risk customers",
            "description": "How often does a low-risk savings customer need to complete a full KYC update?",
            "priority": "medium",
        },
        "customer": {"email": "dhruv.sethi@example.com", "name": "Dhruv Sethi", "company": "Northwind"},
        "expected_answer": "Low-risk customers require a full KYC update every 8 years.",
        "expected_sources": [{"source": "banking-kyc-and-account-update-rules.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "high_risk_kyc_frequency",
        "ticket": {
            "subject": "High-risk customer KYC interval",
            "description": "Please confirm the KYC renewal frequency for high-risk customers.",
            "priority": "medium",
        },
        "customer": {"email": "tara.singh@example.com", "name": "Tara Singh", "company": "Northwind"},
        "expected_answer": "High-risk customers require a full KYC update every 2 years.",
        "expected_sources": [{"source": "banking-kyc-and-account-update-rules.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "accepted_address_proof",
        "ticket": {
            "subject": "Accepted documents for address proof",
            "description": "Which documents count as valid address proof for a profile or KYC update?",
            "priority": "medium",
        },
        "customer": {"email": "anita.sen@example.com", "name": "Anita Sen", "company": "Northwind"},
        "expected_answer": "Accepted address proof includes Aadhaar card, passport, driving license, and a utility bill not older than 3 months.",
        "expected_sources": [{"source": "banking-kyc-and-account-update-rules.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "email_update_timing",
        "ticket": {
            "subject": "How long does an email update take?",
            "description": "If I update my mobile number or email through branch or net banking, how quickly does the change reflect?",
            "priority": "medium",
        },
        "customer": {"email": "rohan.bhat@example.com", "name": "Rohan Bhat", "company": "Northwind"},
        "expected_answer": "Mobile number or email updates require OTP verification and usually reflect within 30 minutes after successful verification.",
        "expected_sources": [{"source": "banking-kyc-and-account-update-rules.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "minor_name_correction",
        "ticket": {
            "subject": "Minor spelling correction in account name",
            "description": "There is a small spelling mistake in my name on the account. How long does a minor correction usually take?",
            "priority": "medium",
        },
        "customer": {"email": "nidhi.rana@example.com", "name": "Nidhi Rana", "company": "Northwind"},
        "expected_answer": "Minor spelling corrections are usually processed in 2 working days.",
        "expected_sources": [{"source": "banking-kyc-and-account-update-rules.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "legal_name_change",
        "ticket": {
            "subject": "Legal name change on my account",
            "description": "I need to update my account after a legal name change. What is required for that process?",
            "priority": "medium",
        },
        "customer": {"email": "gauri.das@example.com", "name": "Gauri Das", "company": "Northwind"},
        "expected_answer": "Legal name changes require supporting legal documents and branch verification.",
        "expected_sources": [{"source": "banking-kyc-and-account-update-rules.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "cheque_activation_after_one_month",
        "ticket": {
            "subject": "When do cheque services start for savings accounts?",
            "description": "I recently opened a savings account. After how long are cheque services activated?",
            "priority": "low",
        },
        "customer": {"email": "vinay.paul@example.com", "name": "Vinay Paul", "company": "Northwind"},
        "expected_answer": "Cheque services for savings account holders are activated only after one month from account opening.",
        "expected_sources": [{"source": "saving-account-rule.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "dmat_not_available",
        "ticket": {
            "subject": "DMAT service on a savings account",
            "description": "Can I enable DMAT service directly on a savings account?",
            "priority": "low",
        },
        "customer": {"email": "hema.nair@example.com", "name": "Hema Nair", "company": "Northwind"},
        "expected_answer": "DMAT service is not available for savings account holders.",
        "expected_sources": [{"source": "saving-account-rule.md", "chunk_index": 0}],
        "expected_tools": [],
    },
    {
        "id": "plan_atm_issue_priority",
        "ticket": {
            "subject": "Enterprise customer asking about an ATM cash reversal",
            "description": "Please draft a reply for a customer asking about an ATM debit without cash and also check if their plan affects the handling priority.",
            "priority": "high",
        },
        "customer": {"email": "priority.atm@example.com", "name": "Priya Rao", "company": "Summit Finance"},
        "expected_answer": "The reply should explain the 24-hour ATM reversal timeline and use the customer plan lookup to decide whether to mention priority handling.",
        "expected_sources": [{"source": "banking-atm-cash-withdrawal-faq.md", "chunk_index": 0}],
        "expected_tools": ["lookup_customer_plan"],
    },
    {
        "id": "plan_kyc_update_priority",
        "ticket": {
            "subject": "Plan-aware response for an email update request",
            "description": "A customer wants to update their email address and asks whether their plan changes the response priority. Please answer the policy part and check the plan.",
            "priority": "high",
        },
        "customer": {"email": "priority.kyc@example.com", "name": "Kabir Joshi", "company": "Summit Finance"},
        "expected_answer": "The reply should mention OTP verification, the 30-minute reflection window, and use the customer plan lookup when describing handling priority.",
        "expected_sources": [{"source": "banking-kyc-and-account-update-rules.md", "chunk_index": 0}],
        "expected_tools": ["lookup_customer_plan"],
    },
    {
        "id": "plan_min_balance_priority",
        "ticket": {
            "subject": "Priority draft for a minimum balance fee question",
            "description": "The customer wants to know the non-maintenance fee on a savings account and also asks if their plan gives them faster support. Please check the plan too.",
            "priority": "high",
        },
        "customer": {"email": "priority.fees@example.com", "name": "Simran Gill", "company": "Summit Finance"},
        "expected_answer": "The reply should mention the typical INR 350 plus taxes non-maintenance fee and use the customer plan lookup before describing response priority.",
        "expected_sources": [{"source": "banking-charges-and-minimum-balance.md", "chunk_index": 0}],
        "expected_tools": ["lookup_customer_plan"],
    },
]

# Keep the committed golden set small enough for reliable free-tier runs while
# still covering the major retrieval domains plus plan/tool-call behavior.
ACTIVE_CASE_IDS = {
    "atm_standard_limit",
    "atm_cash_debited_reversal",
    "atm_wrong_pin_manual_reset",
    "atm_pin_safety_guidance",
    "urban_min_balance",
    "non_maintenance_fee",
    "closure_within_14_days",
    "low_risk_kyc_frequency",
    "accepted_address_proof",
    "email_update_timing",
    "dmat_not_available",
    "plan_atm_issue_priority",
    "plan_kyc_update_priority",
    "plan_min_balance_priority",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the golden eval dataset.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Skip Groq paraphrasing and use the deterministic blueprint wording.",
    )
    return parser.parse_args()


def maybe_llm_refine(case: dict[str, Any], settings: Settings, use_llm: bool) -> dict[str, Any]:
    if not use_llm or not settings.groq_api_key:
        return case

    llm = ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.0,
    )
    system_prompt = (
        "You create concise evaluation cases for a banking support copilot. "
        "Return JSON with keys: subject, description, expected_answer. "
        "Keep every fact grounded in the provided reference answer and do not invent policy details."
    )
    user_prompt = json.dumps(
        {
            "ticket": case["ticket"],
            "expected_answer": case["expected_answer"],
            "expected_sources": case["expected_sources"],
            "expected_tools": case["expected_tools"],
        },
        ensure_ascii=True,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    try:
        payload = json.loads(str(getattr(response, "content", response)))
    except json.JSONDecodeError:
        return case

    return {
        **case,
        "ticket": {
            **case["ticket"],
            "subject": payload.get("subject", case["ticket"]["subject"]),
            "description": payload.get("description", case["ticket"]["description"]),
        },
        "expected_answer": payload.get("expected_answer", case["expected_answer"]),
    }


def build_dataset(seed: int, template_only: bool) -> list[dict[str, Any]]:
    settings = Settings()
    cases = [dict(case) for case in CASE_BLUEPRINTS if case["id"] in ACTIVE_CASE_IDS]
    random.Random(seed).shuffle(cases)
    return [maybe_llm_refine(case, settings=settings, use_llm=not template_only) for case in cases]


def write_dataset(cases: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = build_dataset(seed=args.seed, template_only=args.template_only)
    write_dataset(dataset, args.output)
    print(f"Wrote {len(dataset)} cases to {args.output}")


if __name__ == "__main__":
    main()
