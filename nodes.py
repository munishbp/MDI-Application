"""
Agent nodes for the ticket triage pipeline.
Each node makes a focused LLM call and updates the shared graph state.

Error handling philosophy:
  - Transient failures (malformed JSON, API errors) are retried up to MAX_RETRIES.
  - If retries are exhausted, the node falls back to a safe default AND flags the
    ticket for human review with a specific error message. A silently defaulted
    ticket is a lost ticket.
"""

import json
import os
import time
from anthropic import Anthropic, APIError
from models import TicketCategory, UrgencyLevel, ExtractedEntities, TicketOutput


client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 2
RETRY_DELAY = 1.0  # seconds


def _call_claude(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Shared helper for Claude API calls with retry on transient failures.
    Retries on API errors and empty responses. Raises on exhaustion.
    """
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text.strip()
            if not text:
                raise ValueError("Empty response from API")
            return text

        except (APIError, ValueError, IndexError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue

    raise RuntimeError(f"API call failed after {MAX_RETRIES + 1} attempts: {last_error}")


def _parse_json(raw: str) -> dict:
    """
    Parse JSON from LLM response, handling common issues like
    markdown code fences the model sometimes wraps output in.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]  # drop ```json line
        cleaned = cleaned.rsplit("```", 1)[0]  # drop closing ```
        cleaned = cleaned.strip()

    return json.loads(cleaned)


def _flag_for_review(state: dict, node_name: str, error_msg: str) -> dict:
    """
    Helper to mark a ticket for human review and log the error.
    Returns a dict of state updates to merge.
    """
    errors = list(state.get("processing_errors", []))
    errors.append(f"[{node_name}] {error_msg}")
    return {
        "needs_human_review": True,
        "processing_errors": errors,
    }


# ─── Node 1: Classifier ─────────────────────────────────────────────────────

def classify_ticket(state: dict) -> dict:
    """Classify the inbound ticket into a predefined category."""

    valid_categories = ", ".join([c.value for c in TicketCategory if c != TicketCategory.UNKNOWN])

    system = f"""You are a support ticket classifier for MDI, an insurance software company.
Your job is to classify inbound support tickets into exactly one category.

Valid categories: {valid_categories}

Respond with ONLY a JSON object: {{"category": "<category>"}}
If the ticket does not clearly fit any category, use "Unknown".
Do not include any other text."""

    user = f"Classify this support ticket:\n\n{state['raw_ticket']}"

    try:
        raw = _call_claude(system, user, max_tokens=100)
        parsed = _parse_json(raw)
        category = TicketCategory(parsed["category"])
        return {"category": category}

    except RuntimeError as e:
        # API completely unreachable after retries
        updates = _flag_for_review(state, "classify", f"API failure: {e}")
        updates["category"] = TicketCategory.UNKNOWN
        return updates

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Model returned something unparseable after successful API call
        updates = _flag_for_review(state, "classify", f"Parse failure: {e}")
        updates["category"] = TicketCategory.UNKNOWN
        return updates


# ─── Node 2: Entity Extractor ───────────────────────────────────────────────

def extract_entities(state: dict) -> dict:
    """Extract key entities from the ticket: client name, module, urgency."""

    system = """You are an entity extraction agent for MDI insurance software support.
Extract the following from the support ticket:
- client_name: The name of the client or organization (null if not mentioned)
- module_affected: The software module, feature, or system area affected (null if unclear)
- urgency: Assess urgency as one of: Low, Medium, High, Critical
  - Critical: system down, data loss, blocking all operations
  - High: major feature broken, significant business impact
  - Medium: functionality impaired but workarounds exist
  - Low: cosmetic, enhancement-adjacent, no immediate business impact
- additional_context: Any other relevant details (null if none)

Respond with ONLY a JSON object matching this schema. No other text."""

    user = f"Extract entities from this support ticket:\n\n{state['raw_ticket']}"

    try:
        raw = _call_claude(system, user, max_tokens=300)
        parsed = _parse_json(raw)
        entities = ExtractedEntities(
            client_name=parsed.get("client_name"),
            module_affected=parsed.get("module_affected"),
            urgency=UrgencyLevel(parsed.get("urgency", "Medium")),
            additional_context=parsed.get("additional_context"),
        )
        return {"entities": entities}

    except RuntimeError as e:
        updates = _flag_for_review(state, "extract_entities", f"API failure: {e}")
        updates["entities"] = ExtractedEntities()
        return updates

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        updates = _flag_for_review(state, "extract_entities", f"Parse failure: {e}")
        updates["entities"] = ExtractedEntities()
        return updates


# ─── Node 3: Completeness Checker ───────────────────────────────────────────

def check_completeness(state: dict) -> dict:
    """Determine if the ticket contains enough information to draft a response."""

    # If upstream nodes already failed, don't draft a response based on
    # garbage data. Route to clarification so a human can intervene.
    if state.get("needs_human_review"):
        errors = list(state.get("processing_errors", []))
        errors.append("[check_completeness] Upstream errors detected, defaulting to incomplete")
        return {
            "is_complete": False,
            "processing_errors": errors,
        }

    system = """You are a completeness assessor for insurance software support tickets.
Given a ticket, its classification, and extracted entities, determine whether there is
enough information to draft a meaningful support response.

A ticket is INCOMPLETE if:
- The problem description is vague or ambiguous
- Key reproduction steps are missing for a bug report
- The affected module or feature cannot be identified
- The actual vs expected behavior is unclear
- Critical context is missing that would be needed to help

A ticket is COMPLETE if:
- The issue is clearly described
- Enough context exists to provide a relevant response or next steps

Respond with ONLY a JSON object: {"is_complete": true} or {"is_complete": false}
No other text."""

    entities_str = state["entities"].model_dump_json() if state.get("entities") else "{}"

    user = f"""Ticket: {state['raw_ticket']}
Classification: {state.get('category', 'Unknown')}
Extracted entities: {entities_str}"""

    try:
        raw = _call_claude(system, user, max_tokens=50)
        parsed = _parse_json(raw)
        is_complete = bool(parsed.get("is_complete", False))
        return {"is_complete": is_complete}

    except RuntimeError as e:
        updates = _flag_for_review(state, "check_completeness", f"API failure: {e}")
        updates["is_complete"] = False
        return updates

    except (json.JSONDecodeError, ValueError) as e:
        updates = _flag_for_review(state, "check_completeness", f"Parse failure: {e}")
        updates["is_complete"] = False
        return updates


# ─── Node 4a: Response Drafter (complete tickets) ───────────────────────────

def draft_response(state: dict) -> dict:
    """Draft a suggested support response for a complete ticket."""

    system = """You are a senior support agent at MDI, an insurance software company.
Draft a professional, helpful response to the support ticket.
Your tone should be empathetic but efficient. Include:
- Acknowledgment of the issue
- Concrete next steps or resolution path
- Timeline expectations if applicable

Keep the response under 150 words. Write the response directly -- no JSON wrapping."""

    entities_str = state["entities"].model_dump_json() if state.get("entities") else "{}"

    user = f"""Ticket: {state['raw_ticket']}
Category: {state.get('category', 'Unknown')}
Entities: {entities_str}"""

    try:
        response = _call_claude(system, user, max_tokens=300)
        return {"suggested_response": response}

    except RuntimeError as e:
        updates = _flag_for_review(state, "draft_response", f"API failure: {e}")
        updates["suggested_response"] = None
        return updates


# ─── Node 4b: Clarification Drafter (incomplete tickets) ────────────────────

def draft_clarification(state: dict) -> dict:
    """Draft follow-up questions for an incomplete ticket."""

    system = """You are a senior support agent at MDI, an insurance software company.
The support ticket you received does not contain enough information to resolve the issue.
Draft a professional follow-up message requesting the specific missing information.

Be precise about what you need. Ask targeted questions rather than generic ones.
Keep it under 120 words. Write the response directly -- no JSON wrapping."""

    entities_str = state["entities"].model_dump_json() if state.get("entities") else "{}"

    user = f"""Ticket: {state['raw_ticket']}
Category: {state.get('category', 'Unknown')}
Entities: {entities_str}
What information is missing or unclear that prevents you from resolving this?"""

    try:
        response = _call_claude(system, user, max_tokens=250)
        return {"clarification_request": response}

    except RuntimeError as e:
        updates = _flag_for_review(state, "draft_clarification", f"API failure: {e}")
        updates["clarification_request"] = None
        return updates


# ─── Node 5: Output Assembler ───────────────────────────────────────────────

def assemble_output(state: dict) -> dict:
    """Assemble the final structured TicketOutput from accumulated state."""

    needs_review = state.get("needs_human_review", False)
    errors = list(state.get("processing_errors", []))

    # Final safety net: if we have no response AND no clarification,
    # something went wrong that wasn't caught — flag it.
    if not state.get("suggested_response") and not state.get("clarification_request"):
        needs_review = True
        errors.append("[assemble_output] No response or clarification generated — ticket requires manual handling")

    output = TicketOutput(
        category=state.get("category", TicketCategory.UNKNOWN),
        entities=state.get("entities", ExtractedEntities()),
        is_complete=state.get("is_complete", False),
        suggested_response=state.get("suggested_response"),
        clarification_request=state.get("clarification_request"),
        needs_human_review=needs_review,
        processing_errors=errors,
    )

    return {"output": output}
