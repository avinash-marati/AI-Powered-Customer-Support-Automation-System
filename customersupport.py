# =============================================================
# ASSIGNMENT 2 — AI-Powered Customer Support Automation System
# Built with LangGraph
# =============================================================

import os
import sqlite3
import re
from typing import Annotated, TypedDict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages        
from langgraph.checkpoint.sqlite import SqliteSaver     
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

llm = ChatOllama(model="qwen2.5:3b", temperature=0)

DB_PATH = "memory.db"


# =============================================================
# TASK 2 — STATE STRUCTURE
# SqliteSaver saves/restores this entire state automatically
# =============================================================

class SupportState(TypedDict):
    # Core query fields
    query: str
    customer_name: Optional[str]
    department: str
    answer: str

    #RAG context retrieved from documents
    rag_context: str

    # Task 7 — Conversation memory via LangGraph message accumulator
    messages: Annotated[list, add_messages]

    # Task 8 — Human-in-the-Loop approval
    requires_approval: bool
    approval_status: str        # "not_required" | "approved" | "rejected"
    escalation_reason: str

    # Task 9 — Supervisor review
    supervisor_feedback: str
    final_response: str


# =============================================================
# TASK 6 — RAG PIPELINE  (keyword scoring over .txt documents)
# =============================================================

DOCUMENTS = {}

def load_documents():
    """Load company knowledge base documents from disk at startup."""
    doc_files = {
        "Company_policy":   "Company_policy.txt",
        "Pricing_guide":    "Pricing_guide.txt",
        "Technical_manual": "Technical_manual.txt",
        "FAQ_document":     "FAQ_document.txt",
    }
    for name, filename in doc_files.items():
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                DOCUMENTS[name] = f.read()
            console.print(f"[dim]Loaded: {filename}[/dim]")
        else:
            console.print(f"[yellow]Warning: {filename} not found.[/yellow]")


def chunk_document(text: str, chunk_size: int = 300) -> list:
    """Split document text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def rag_retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant document chunks for a query.
    Scores each chunk by how many unique query keywords it contains.
    Returns the top_k chunks labelled with their source document.
    """
    stop_words = {"i", "my", "the", "a", "an", "is", "are", "was", "what",
                  "how", "can", "do", "does", "for", "to", "in", "of", "and",
                  "or", "me", "it", "this", "that", "with", "have", "has"}
    query_words = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split()) - stop_words

    scored = []
    for doc_name, doc_text in DOCUMENTS.items():
        for chunk in chunk_document(doc_text):
            score = sum(1 for w in query_words if w in chunk.lower())
            if score > 0:
                scored.append((score, chunk, doc_name))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return "No relevant information found in the knowledge base."

    parts = [f"[Source: {doc}]\n{chunk.strip()}" for _, chunk, doc in top]
    return "\n\n---\n\n".join(parts)


# =============================================================
# TASK 3 — INTENT CLASSIFICATION NODE
# =============================================================

def classify_node(state: SupportState) -> SupportState:
    """
    Classifies the customer query into one of:
    sales | technical | billing | account | memory

    Also detects high-risk phrases that require human approval (Task 8).
    """
    prompt = f"""You are a customer support classifier.
Classify the customer query into EXACTLY one category from this list:
sales, technical, billing, account, memory

Rules:
- Reply with ONLY one word from the list above. Nothing else.
- No punctuation, no explanation, no extra words.
- sales     = product information, subscription plans, pricing details
- technical = application errors, installation issues, login problems, configuration issues
- billing   = invoice requests, payment issues, refund requests
- account   = password reset, profile updates, account activation/deactivation
- memory    = customer asking about their previous issue, prior conversation, or past query

Query: {state['query']}

Your answer (one word only):"""

    result = llm.invoke(prompt)
    raw = result.content.strip().lower().strip(".")

    valid = ["sales", "technical", "billing", "account", "memory"]
    department = raw if raw in valid else "general"

    # Extract customer name if mentioned
    name_match = re.search(r"my name is ([A-Za-z]+)", state["query"], re.IGNORECASE)
    customer_name = name_match.group(1) if name_match else state.get("customer_name", "")

    # Detect high-risk phrases → triggers human-in-the-loop (Task 8)
    query_lower = state["query"].lower()
    escalation_map = {
        "refund":                "Customer is requesting a refund.",
        "cancel subscription":   "Customer wants to cancel their subscription.",
        "close account":         "Customer wants to permanently close their account.",
        "compensation":          "Customer is requesting compensation.",
        "escalate":              "Customer is requesting management escalation.",
    }
    requires_approval = False
    escalation_reason = ""
    for keyword, reason in escalation_map.items():
        if keyword in query_lower:
            requires_approval = True
            escalation_reason = reason
            break

    console.print(f"[bold cyan]  Classified as: {department.upper()}[/bold cyan]")
    if requires_approval:
        console.print(f"[bold red]  ⚠  Escalation required: {escalation_reason}[/bold red]")

    return {
        **state,
        "department":        department,
        "customer_name":     customer_name,
        "requires_approval": requires_approval,
        "escalation_reason": escalation_reason,
        "approval_status":   "pending" if requires_approval else "not_required",
    }


# =============================================================
# TASK 4 — CONDITIONAL ROUTING
# =============================================================

def route_query(state: SupportState) -> str:
    """Route to the correct department node (or human approval for high-risk)."""
    if state.get("requires_approval", False):
        return "human_approval"
    route_map = {
        "sales":     "sales_agent",
        "technical": "technical_agent",
        "billing":   "billing_agent",
        "account":   "account_agent",
        "memory":    "memory_agent",
    }
    return route_map.get(state.get("department", ""), "general_agent")


# =============================================================
# TASK 5 — SPECIALIZED SUPPORT AGENTS  (RAG-enhanced, Task 6)
# =============================================================

def _rag_agent(state: SupportState, dept_name: str, dept_scope: str) -> SupportState:
    """
    Generic agent used by all department nodes.
    Retrieves relevant knowledge-base chunks via RAG, then uses the LLM
    to generate a professional customer response grounded in those chunks.
    """
    rag_context = rag_retrieve(state["query"])

    prompt = f"""You are the {dept_name} support agent at ABC Technologies.
{dept_scope}

Use the following knowledge base context to answer the customer query accurately:

--- KNOWLEDGE BASE CONTEXT ---
{rag_context}
--- END CONTEXT ---

Customer Query: {state['query']}

Provide a clear, helpful, and professional response using the context above.
If the answer is not in the context, acknowledge the query politely and ask for more details.

Response:"""

    result = llm.invoke(prompt)

    return {
        **state,
        "rag_context": rag_context,
        "answer":      result.content.strip(),
    }


def sales_agent(state: SupportState) -> SupportState:
    return _rag_agent(
        state,
        "Sales Support",
        "You handle product information, subscription plans, and pricing details."
    )


def technical_agent(state: SupportState) -> SupportState:
    return _rag_agent(
        state,
        "Technical Support",
        "You handle application errors, installation issues, login problems, and configuration issues."
    )


def billing_agent(state: SupportState) -> SupportState:
    return _rag_agent(
        state,
        "Billing Support",
        "You handle invoice requests, payment issues, and refund requests."
    )


def account_agent(state: SupportState) -> SupportState:
    return _rag_agent(
        state,
        "Account Support",
        "You handle password resets, profile updates, and account activation/deactivation."
    )


def general_agent(state: SupportState) -> SupportState:
    return {
        **state,
        "rag_context": "",
        "answer": (
            "Thank you for contacting ABC Technologies Support.\n"
            "Our team will get back to you shortly.\n"
            "Operating hours: Mon–Sat, 9 AM – 6 PM IST.\n"
            "Reference: " + state["query"][:30] + "..."
        ),
    }


# =============================================================
# TASK 7 — MEMORY AGENT  (reads LangGraph checkpointed messages)
# =============================================================

def memory_agent(state: SupportState) -> SupportState:
    """
    Answers questions about previous interactions by reading the
    message history that SqliteSaver has already restored into
    state["messages"] for this thread_id.
    """
    # Build a readable history string from the accumulated messages
    history_text = ""
    for msg in state["messages"][:-1]:   # exclude the current query
        if hasattr(msg, "content"):
            role = "Customer" if isinstance(msg, HumanMessage) else "Support"
            history_text += f"{role}: {msg.content}\n"

    if not history_text.strip():
        history_text = "No previous conversation history found for this session."

    prompt = f"""You are a helpful customer support assistant with access to this
customer's conversation history from the current session.

Conversation history:
{history_text}

Customer's current question: {state['query']}

Answer the customer's question using the conversation history above.
If there is no relevant history, let them know politely.

Response:"""

    result = llm.invoke(prompt)

    return {
        **state,
        "rag_context": history_text,
        "answer":      result.content.strip(),
        "department":  "memory",
    }


# =============================================================
# TASK 8 — HUMAN-IN-THE-LOOP APPROVAL NODE
# =============================================================

def human_approval_node(state: SupportState) -> SupportState:
    """
    Pauses execution and asks a human supervisor to approve or reject
    the high-risk request before a response is generated.
    """
    console.print("\n" + "=" * 60)
    console.print("[bold red]⚠   SUPERVISOR APPROVAL REQUIRED[/bold red]")
    console.print("=" * 60)
    console.print(f"[yellow]Query:[/yellow]             {state['query']}")
    console.print(f"[yellow]Escalation reason:[/yellow] {state['escalation_reason']}")
    console.print(f"[yellow]Department:[/yellow]        {state['department'].upper()}")

    # Show relevant policy context to help the supervisor decide
    policy_context = rag_retrieve(state["query"])
    console.print(f"\n[dim]Relevant policy snippet:\n{policy_context[:400]}[/dim]")

    console.print("")
    decision = Prompt.ask(
        "[bold]Supervisor: approve or reject this request?[/bold]",
        choices=["approve", "reject"],
        default="approve"
    )

    approval_status = "approved" if decision == "approve" else "rejected"
    console.print(f"[green]  → Decision recorded: {approval_status.upper()}[/green]\n")

    return {
        **state,
        "rag_context":   policy_context,
        "approval_status": approval_status,
    }


def route_after_approval(state: SupportState) -> str:
    """After supervisor review, route to the department agent or rejection handler."""
    if state.get("approval_status") == "approved":
        dept = state.get("department", "billing")
        return {
            "sales":     "sales_agent",
            "technical": "technical_agent",
            "billing":   "billing_agent",
            "account":   "account_agent",
        }.get(dept, "billing_agent")
    return "rejection_agent"


def rejection_agent(state: SupportState) -> SupportState:
    """Generates a polite refusal when the supervisor rejects the request."""
    return {
        **state,
        "answer": (
            "Thank you for contacting ABC Technologies.\n\n"
            "After careful review, we are unable to process your request at this time.\n"
            "If you believe this is an error, please contact our support team directly:\n"
            "  • Email: support@abctech.com\n"
            "  • Phone: Mon–Fri, 9 AM – 6 PM IST\n\n"
            f"Reference query: {state['query'][:40]}..."
        ),
    }


# =============================================================
# TASK 9 — SUPERVISOR AGENT  (response quality check)
# =============================================================

def supervisor_agent(state: SupportState) -> SupportState:
    """
    Reviews the draft answer from the department agent.
    Approves it as-is if it is good, or rewrites it to meet quality standards.
    This is the last node before the response reaches the customer.
    """
    draft = state.get("answer", "")

    prompt = f"""You are a Quality Assurance Supervisor at ABC Technologies.
Review the following AI-generated customer support response.

Customer Query: {state['query']}

Draft Response:
{draft}

Your job:
1. Check if the response directly addresses the customer's query.
2. Check if the tone is professional and polite.
3. Check if any knowledge-base information has been used correctly.
4. If the response is good, output it exactly as-is.
5. If it needs improvement, rewrite it to be better.

Output ONLY the final customer-facing response. No commentary, no labels.

Final Response:"""

    result = llm.invoke(prompt)

    return {
        **state,
        "supervisor_feedback": "Response reviewed and approved by supervisor agent.",
        "final_response":      result.content.strip(),
    }


# =============================================================
# TASK 1 — LANGGRAPH WORKFLOW  (build_graph)
# =============================================================

def build_graph():
    """
    Build and compile the full LangGraph workflow with SqliteSaver
    checkpointing for persistent SQLite memory (Lab 6 pattern).
    """
    builder = StateGraph(SupportState)

    # Register all nodes
    builder.add_node("classify",        classify_node)
    builder.add_node("sales_agent",     sales_agent)
    builder.add_node("technical_agent", technical_agent)
    builder.add_node("billing_agent",   billing_agent)
    builder.add_node("account_agent",   account_agent)
    builder.add_node("general_agent",   general_agent)
    builder.add_node("memory_agent",    memory_agent)
    builder.add_node("human_approval",  human_approval_node)
    builder.add_node("rejection_agent", rejection_agent)
    builder.add_node("supervisor",      supervisor_agent)

    # Entry point
    builder.set_entry_point("classify")

    # Task 4 — Conditional routing from classifier
    builder.add_conditional_edges(
        "classify",
        route_query,
        {
            "sales_agent":     "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent":   "billing_agent",
            "account_agent":   "account_agent",
            "general_agent":   "general_agent",
            "memory_agent":    "memory_agent",
            "human_approval":  "human_approval",
        }
    )

    # Task 8 — Conditional routing after human approval
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "sales_agent":     "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent":   "billing_agent",
            "account_agent":   "account_agent",
            "rejection_agent": "rejection_agent",
        }
    )

    # All department agents → supervisor
    for node in ["sales_agent", "technical_agent", "billing_agent",
                 "account_agent", "general_agent", "memory_agent", "rejection_agent"]:
        builder.add_edge(node, "supervisor")

    # Supervisor → END
    builder.add_edge("supervisor", END)

    # ── TASK 7: Attach SqliteSaver checkpointer ──
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)
    graph = builder.compile(checkpointer=memory)
    return graph


# =============================================================
# TASK 10 — DEMONSTRATION WITH 5 SAMPLE QUERIES
# =============================================================

DEMO_QUERIES = [
    "What are the pricing plans available for your software?",  # → Sales
    "I forgot my account password.",                            # → Account
    "My application crashes whenever I upload a file.",         # → Technical
    "I need a refund for my annual subscription.",              # → Billing (human approval)
    "What was my previous support issue?",                      # → Memory recall
]


def run_demo():
    """
    Run all 5 sample queries in sequence using a single thread_id so
    that Query 5 (memory recall) can see the history from Queries 1–4.
    """
    thread_id = "demo_customer_001"
    config = {"configurable": {"thread_id": thread_id}}

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  AI Customer Support — DEMO MODE (Task 10)[/bold magenta]")
    console.print(f"[bold magenta]  Thread ID: {thread_id}[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    graph = build_graph()

    for i, query in enumerate(DEMO_QUERIES, 1):
        console.print(f"\n[bold white]━━━  Query {i} / 5  ━━━[/bold white]")
        console.print(f"[bold yellow]Customer:[/bold yellow] {query}\n")

        # Each invoke carries the current message.
        # SqliteSaver automatically merges it with stored history (add_messages).
        initial_state: SupportState = {
            "query":             query,
            "customer_name":     "",
            "department":        "",
            "answer":            "",
            "rag_context":       "",
            "messages":          [HumanMessage(content=query)],
            "requires_approval": False,
            "approval_status":   "not_required",
            "escalation_reason": "",
            "supervisor_feedback": "",
            "final_response":    "",
        }

        result = graph.invoke(initial_state, config=config)

        dept_label = result.get("department", "general").upper()
        final = result.get("final_response") or result.get("answer", "No response generated.")

        console.print(Panel(
            final,
            title=f"[bold green]Response — {dept_label}[/bold green]",
            border_style="green",
        ))

        if result.get("approval_status") not in ("not_required", ""):
            color = "green" if result["approval_status"] == "approved" else "red"
            console.print(f"[{color}]Approval Status: {result['approval_status'].upper()}[/{color}]")

    console.print("\n[bold yellow]All 5 queries processed successfully.[/bold yellow]")
    console.print(f"[dim]Memory saved to: {os.path.abspath(DB_PATH)}[/dim]\n")


def run_interactive():
    """
    Interactive chat loop.
    All conversations are saved per thread_id by SqliteSaver automatically.
    """
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  AI Customer Support — Interactive Mode[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    thread_id_input = console.input(
        "\n[cyan]Enter Thread ID (press Enter for 'user_001'): [/cyan]"
    ).strip()
    thread_id = thread_id_input if thread_id_input else "user_001"

    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"\n[green]Using thread: [bold]{thread_id}[/bold][/green]")
    console.print("[dim]Type 'quit' to exit. Conversation is saved automatically.[/dim]\n")

    graph = build_graph()

    while True:
        user_input = console.input("[bold cyan]You: [/bold cyan]").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            console.print(f"[yellow]Conversation saved to {DB_PATH}. See you next time![/yellow]")
            break

        initial_state: SupportState = {
            "query":             user_input,
            "customer_name":     "",
            "department":        "",
            "answer":            "",
            "rag_context":       "",
            "messages":          [HumanMessage(content=user_input)],
            "requires_approval": False,
            "approval_status":   "not_required",
            "escalation_reason": "",
            "supervisor_feedback": "",
            "final_response":    "",
        }

        result = graph.invoke(initial_state, config=config)
        final = result.get("final_response") or result.get("answer", "No response generated.")
        dept_label = result.get("department", "general").upper()

        console.print(Panel(
            final,
            title=f"[bold green]Assistant ({dept_label}) — thread: {thread_id}[/bold green]",
            border_style="green",
        ))

    console.print(f"\n[dim]Memory file: {os.path.abspath(DB_PATH)}[/dim]")


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    load_documents()

    console.print("\n[bold cyan]Choose mode:[/bold cyan]")
    console.print("  [white]1[/white] — Demo mode (5 sample queries, Task 10)")
    console.print("  [white]2[/white] — Interactive mode (type your own queries)")

    choice = console.input("\n[bold cyan]Enter 1 or 2: [/bold cyan]").strip()

    if choice == "1":
        run_demo()
    else:
        run_interactive()
