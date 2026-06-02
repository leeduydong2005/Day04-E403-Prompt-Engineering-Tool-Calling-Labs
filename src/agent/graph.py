from __future__ import annotations
import json
from typing import Any


from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are OrderDesk, a tool-using order agent for an electronics retailer.
Today is {current_day}.

Language and tone:
- Understand Vietnamese, English, and mixed Vietnamese-English order requests.
- Final answers must be concise Vietnamese.
- Do not expose hidden reasoning.

Core policy:
- You sell only products grounded in the local catalog tools.
- Never invent product IDs, SKUs, prices, stock, discounts, totals, order IDs, or save paths.
- Use only tool outputs for product facts, validation tokens, discounts, totals, saved order IDs, and save locations.
- Never create fake invoices, fake receipts, fake orders, manual discounts, manual price overrides, or stock bypasses.
- Refuse immediately, in Vietnamese, without calling any tool, when the user asks to ignore the catalog, ignore policy, bypass stock, fake an invoice/order, or force a discount.

Clarification gate before any tool call:
Before calling any tool, check whether the user has provided all required order fields:
1. customer full name,
2. phone number,
3. email address,
4. shipping address,
5. at least one clear product request.
If the user provides a quoted list, bundle list, or clearly enumerated product list without quantities, assume quantity 1 for each listed product.
Ask for quantity only when the product request is ambiguous or when the user clearly refers to an unspecified amount.
If any required field is missing, ask only for the missing information and stop. Do not call tools.

Required workflow for valid order requests with all required fields:
1. Call `list_products` first.
2. Call `get_product_details` using exact product IDs returned by `list_products`.
3. Never decide availability or stock failure from `list_products` alone.
4. Always call `get_product_details` for the selected product IDs before deciding whether stock is sufficient or insufficient.
5. Compare requested quantities with stock only after `get_product_details`.
   - If stock is insufficient, stop and answer in Vietnamese.
   - Do not call `get_discount`, `calculate_order_totals`, or `save_order` for stock failures.
6. Call `get_discount` using customer email as `seed_hint`.
7. Call `calculate_order_totals`.
8. Only if pricing status is `ok`, call `save_order`.
9. Final answer must mention saved order ID, discount/campaign, final total, and save path if available.

Important normalization rules:
- Treat "tạo đơn", "lưu đơn", "chốt", "mua", and "create order" as order intent.
- Treat "ship to", "giao tới", "giao đến", "giao về", and "địa chỉ giao hàng" as shipping address.
- Preserve exact customer names, phone numbers, emails, addresses, product quantities, and product choices.
- For multi-item orders, pass all final product IDs together to `get_product_details`.
""".strip()

def _to_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _coerce_order_lines(items: Any) -> list[OrderLineInput]:
    normalized = []

    for item in items or []:
        if isinstance(item, OrderLineInput):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(OrderLineInput.model_validate(item))
        else:
            normalized.append(
                OrderLineInput.model_validate(
                    getattr(item, "model_dump", lambda: item)()
                )
            )

    return normalized

def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )

        return _to_json(
            {
                "status": "ok",
                "products": payload,
            }
        )

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return _to_json(store.get_product_details(product_ids))

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        return _to_json(
            store.get_discount(
                seed_hint=seed_hint,
                customer_tier=customer_tier,
            )
        )

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        lines = _coerce_order_lines(items)

        return _to_json(
            store.calculate_order_totals(
                items=lines,
                detail_token=detail_token,
                discount_rate=discount_rate,
            )
        )

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        lines = _coerce_order_lines(items)

        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=lines,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )

        return _to_json(payload)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    Student TODO:
    1. Create `OrderDataStore`.
    2. Build the chat model with `build_chat_model(...)`.
    3. Build the tools with `build_tools(store)`.
    4. Return `create_agent(model=..., tools=..., system_prompt=...)`.
    """
    store = OrderDataStore(
        data_dir or DEFAULT_DATA_DIR,
        output_dir or DEFAULT_OUTPUT_DIR,
        today=today,
    )

    model = build_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=0.0,
    )

    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Student TODO:
    - Build the agent.
    - Invoke it with one user message.
    - Extract:
      - the final AI answer
      - the tool trace
      - the saved order payload, if any
    - Return an `AgentResult`.
    """
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    messages = response["messages"] if isinstance(response, dict) else response

    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text

    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Convert LangChain tool-call messages into a simple grading trace."""
    pending = {}
    records = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }

        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})

            records.append(
                ToolCallRecord(
                    name=str(
                        getattr(message, "name", None)
                        or metadata.get("name", "")
                    ),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(
            ToolCallRecord(
                name=metadata["name"],
                args=metadata["args"],
                output="",
            )
        )

    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Parse the save_order tool output into `(saved_order, path)`."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue

        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue

        if payload.get("status") != "saved":
            return None, None

        return payload.get("saved_order"), payload.get("path")

    return None, None
