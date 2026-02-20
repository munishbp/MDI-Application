"""
CLI entry point for the MDI Ticket Triage Agent.

Usage:
    python main.py                          # Run all test cases
    python main.py --ticket "your ticket"   # Run a single ticket
    python main.py --file ticket.txt        # Run from file
"""

import argparse
import json
import sys
import time
from graph import triage_agent
from models import TicketOutput


def process_ticket(ticket_text: str) -> TicketOutput:
    """Run a single ticket through the triage agent and return structured output."""

    initial_state = {
        "raw_ticket": ticket_text,
        "category": None,
        "entities": None,
        "is_complete": None,
        "suggested_response": None,
        "clarification_request": None,
        "needs_human_review": False,
        "processing_errors": [],
        "output": None,
    }

    final_state = triage_agent.invoke(initial_state)
    return final_state["output"]


def print_result(ticket: str, output: TicketOutput, elapsed: float) -> None:
    """Pretty-print a triage result."""

    print("=" * 70)
    print("TICKET:")
    print(ticket.strip())
    print("-" * 70)
    print("STRUCTURED OUTPUT:")
    print(json.dumps(output.model_dump(mode="json"), indent=2))
    print(f"\n[Processed in {elapsed:.1f}s]")
    print("=" * 70)
    print()


# ─── Built-in Test Cases ────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "Test 1: Billing Dispute",
        "ticket": """Subject: Incorrect invoice - URGENT

Hi Support,

This is Maria Chen from Lakewood Mutual Insurance. We received our January
invoice and the amount is $14,200 — nearly double what we normally pay.
We haven't added any new users or modules since last quarter. Our account
number is LM-40291.

Can someone look into this immediately? Our accounts payable team needs
to process this by end of week.

Thanks,
Maria Chen
VP of Operations, Lakewood Mutual Insurance""",
    },
    {
        "name": "Test 2: Vague Bug Report (Incomplete)",
        "ticket": """Subject: Something is broken

Hey,

The system is acting weird again. When I try to do the thing with the
claims it just doesn't work right. It was fine last week I think.

Can you fix this?

- Tom""",
    },
    {
        "name": "Test 3: System Incompatibility",
        "ticket": """Subject: PDF export fails after Chrome 132 update

Support Team,

Since our organization updated to Chrome 132.0.6834 last Tuesday, the
policy document PDF export in the PolicyAdmin module throws a
"Renderer Process Crashed" error every time. We've confirmed the issue
on 15+ workstations running Windows 11 23H2. Reverting to Chrome 131
resolves it. Firefox and Edge are unaffected.

We need this resolved ASAP as our compliance team generates 200+ policy
PDFs daily and the Chrome rollback is not sustainable.

Contact: David Park, IT Director, Meridian Risk Group
Ticket ref: INT-2024-0892""",
    },
]


def run_test_cases() -> None:
    """Run all built-in test cases and display results."""

    print("\n" + "=" * 70)
    print("MDI TICKET TRIAGE AGENT — TEST SUITE")
    print("=" * 70 + "\n")

    for i, case in enumerate(TEST_CASES):
        print(f"\n>>> {case['name']}")
        start = time.time()
        try:
            output = process_ticket(case["ticket"])
            elapsed = time.time() - start
            print_result(case["ticket"], output, elapsed)
        except Exception as e:
            print(f"  ERROR: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="MDI Ticket Triage Agent")
    parser.add_argument("--ticket", type=str, help="Process a single ticket (inline text)")
    parser.add_argument("--file", type=str, help="Process a ticket from a text file")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            ticket_text = f.read()
        start = time.time()
        output = process_ticket(ticket_text)
        elapsed = time.time() - start
        print_result(ticket_text, output, elapsed)
    elif args.ticket:
        start = time.time()
        output = process_ticket(args.ticket)
        elapsed = time.time() - start
        print_result(args.ticket, output, elapsed)
    else:
        run_test_cases()


if __name__ == "__main__":
    main()
