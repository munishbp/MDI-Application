# MDI Support Ticket Triage Agent

An AI-powered support ticket triage system built for MDI's insurance software platform. The agent classifies inbound tickets, extracts key entities, assesses completeness, and either drafts a response or requests clarification — all through a structured, branching workflow.

## Architecture

### Why LangGraph Over a Single LLM Call

A single prompt can classify, extract, and draft simultaneously. For many use cases that's the right choice: fewer API calls, lower latency, simpler code. I chose a multi-node graph for two reasons specific to this problem:

1. **Conditional routing.** Not every ticket has enough information to respond to. Vague bug reports, incomplete descriptions, and ambiguous requests need follow-up questions. This creates a fork in the workflow: assess completeness, then branch. LangGraph's conditional edges express this in a cleaner format. A linear pipeline or single prompt cannot route execution based on intermediate reasoning without external control flow bolted on after the fact.

2. **Node-level debuggability.** In a production triage system, you need to know *where* the agent went wrong. Was the category correct but the urgency assessment off? Did entity extraction miss the client name? Isolated nodes with focused prompts make each step independently testable, loggable, and refinable. This matters when you're iterating on prompt quality across hundreds of real tickets.

The tradeoff is latency (4-5 sequential API calls vs. 1) and cost. For a triage system where tickets arrive asynchronously and response time is not expected instantly, this is acceptable. For a real-time chatbot, it would not be.

### Graph Topology

```
[Classifier] → [Entity Extractor] → [Completeness Check]
                                            ↓
                                   ┌────────┴────────┐
                                   ↓                 ↓
                             [Complete]         [Incomplete]
                                   ↓                 ↓
                          [Response Drafter]  [Clarification Drafter]
                                   ↓                 ↓
                              [Output Assembler] ←───┘
```

Each node has a single responsibility:

| Node | Role | Output |
|------|------|--------|
| `classify_ticket` | Maps ticket to one of 7 categories using enum-constrained prompting | `TicketCategory` |
| `extract_entities` | Pulls client name, affected module, urgency level, and context | `ExtractedEntities` |
| `check_completeness` | Assesses whether enough information exists to draft a response | `bool` |
| `draft_response` | Writes a professional support response for complete tickets | `str` |
| `draft_clarification` | Writes targeted follow-up questions for incomplete tickets | `str` |
| `assemble_output` | Packages all accumulated state into the final JSON structure | `TicketOutput` |

### Error Handling

The original version of this agent silently substituted defaults when parsing failed, `Unknown` category, `Medium` urgency, then kept going. The problem: a ticket that hits a parse error comes out the other end looking like a successfully processed ticket. In a triage system, a silently defaulted ticket is a lost ticket.

The revised approach distinguishes two failure modes:

**Transient failures** (API timeouts, malformed JSON from model hiccups) are retried up to 2 additional times with linear backoff. These usually succeed on retry because the same input produces valid output on a second attempt.

**Exhausted retries** trigger a fallback to safe defaults, but critically, the ticket is also flagged with `needs_human_review: true` and a specific error message in `processing_errors`. The ticket still flows through the pipeline (so it doesn't disappear), but the output is honest about what went wrong.

Additional safeguards:
- If upstream nodes failed, the completeness checker automatically routes to the clarification branch rather than drafting a response on garbage data.
- The output assembler performs a final safety check: if neither a response nor a clarification was generated, the ticket is flagged for manual handling regardless of what upstream nodes reported.
- The `_parse_json` helper strips markdown code fences that LLMs occasionally wrap JSON output in, reducing false parse failures.

## Setup

```bash
# Clone and install
git clone <repo-url>
cd mdi-ticket-agent
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

```bash
# Run all 3 built-in test cases
python main.py

# Process a single ticket inline
python main.py --ticket "Our claims module is throwing 500 errors on submission"

# Process from file
python main.py --file path/to/ticket.txt
```

## Test Cases

### Test 1: Billing Dispute
A clear, well documented billing discrepancy from an identified client with specific dollar amounts and account reference. Tests the agent's ability to classify accurately and draft a concrete response with next steps.

**Expected behavior:** Classified as `Billing`, marked complete, response drafted with acknowledgment and escalation path.

### Test 2: Vague Bug Report (Incomplete)
An intentionally underspecified ticket ie. no client name, no module identified, no reproduction steps, no error details. Just "the system is acting weird."

**Expected behavior:** Classified as `Bug`, marked incomplete, clarification request drafted asking for specific module, reproduction steps, error messages, and browser/environment details.

### Test 3: System Incompatibility
A well-documented compatibility issue between a browser update and a specific MDI module, with affected workstation count, OS version, and confirmed workaround (rollback). Tests the agent's handling of infrastructure level issues that don't fit neatly into the standard Bug category.

**Expected behavior:** Classified as `System Incompatibility` or `Bug`, marked complete, response drafted acknowledging the Chrome 132 conflict and outlining investigation steps.

## Project Structure

```
mdi-ticket-agent/
├── main.py              # CLI entry point and test cases
├── graph.py             # LangGraph workflow definition
├── nodes.py             # Agent node functions (LLM calls)
├── state.py             # Graph state TypedDict
├── models.py            # Pydantic models (entities, output schema)
├── requirements.txt
└── README.md
```

## Potential Extensions

If this were moving toward production, the next steps would be:

- **Confidence scoring.** Each node could output a confidence value, enabling automatic escalation to human review when the agent is uncertain.
- **Feedback loop.** Log agent decisions alongside human corrections to build evaluation datasets for prompt refinement.
- **JIRA integration.** The `TicketOutput` JSON maps cleanly to JIRA issue creation via API — category becomes label, urgency becomes priority, entities populate custom fields.
- **Batch processing.** LangGraph supports async execution; processing ticket queues in parallel would reduce throughput bottleneck.
