"""Оценка качества поиска норм: прогоняет каждое описание из gold-датасета
через граф агента до узла retrieve_norms, сравнивает выдачу с эталонными
ссылками и пишет метрики (precision@k, recall@k, hit@k, latency) в CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = BASE_DIR if (BASE_DIR / "config.py").exists() else BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from config import settings  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agent.graph import build_state_graph  # noqa: E402

LOGGER = logging.getLogger("scripts.eval_retrieval")

_FILENAME_SAFE_RE = re.compile(r"[^\w.-]+", re.UNICODE)
_USER_SIMULATOR_PROMPT = (
    "Ты симулируешь поведение пользователя в чате с юридическим ассистентом.\n\n"
    "Исходное описание сделки от пользователя:\n{initial_request}\n\n"
    "Ассистент задаёт уточняющий вопрос:\n{question}\n\n"
    "Ответь как пользователь — 1-3 коротких предложения. Если в исходном описании\n"
    "уже есть информация для ответа, используй её. Если нет — придумай реалистичный\n"
    "ответ, согласующийся с описанием. Верни только текст ответа без префиксов\n"
    "вроде «Ответ:» и без объяснений."
)


def _load_gold_set(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return data["entries"]
    raise ValueError(f"Unsupported gold-set shape in {path}: expected list or {{'entries': [...]}}")


def _make_user_simulator() -> ChatOllama:
    return ChatOllama(
        model=settings.reference_llm_model or settings.llm_model,
        base_url=settings.reference_llm_base_url or settings.ollama_base_url,
        temperature=0.4,
        num_ctx=2048,
        num_predict=256,
        reasoning=settings.reference_llm_reasoning,
    )


def _simulate_user_answer(simulator: ChatOllama, initial_request: str, question: str) -> str:
    prompt = _USER_SIMULATOR_PROMPT.format(initial_request=initial_request, question=question)
    response = simulator.invoke([HumanMessage(content=prompt)])
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content.strip()


def _compile_eval_graph():
    graph = build_state_graph()
    return graph.compile(checkpointer=MemorySaver(), interrupt_after=["retrieve_norms"])


def _make_initial_state(description: str) -> dict:
    return {
        "messages": [HumanMessage(content=description)],
        "deal_description": description,
        "ask_personal_data": False,
        "contract_generation_policy": "always",
        "allow_edit": True,
        "user_plan": "eval",
        "max_iterations": settings.max_iterations,
        "max_classification_clarifications": settings.max_classification_clarifications,
    }


def _extract_clarification_question(state: dict) -> str | None:
    if not state.get("clarification_needed"):
        return None
    q = state.get("clarification_question")
    return q.strip() if isinstance(q, str) and q.strip() else None


def _run_agent_until_retrieve(
    app, simulator: ChatOllama, initial_request: str, max_clarifications: int
) -> tuple[list[dict], str | None, float | None]:
    description = initial_request
    last_classified: str | None = None
    for round_idx in range(max_clarifications + 1):
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = app.invoke(_make_initial_state(description), config=config)
        if state.get("deal_type"):
            last_classified = state.get("deal_type")

        retrieved = state.get("retrieved_norms")
        if retrieved:
            latency = state.get("retrieve_norms_latency_ms")
            LOGGER.info(
                "round=%s reached retrieve_norms with %s chunks deal_type=%r latency=%.1fms",
                round_idx,
                len(retrieved),
                state.get("deal_type"),
                latency or 0.0,
            )
            return retrieved, state.get("deal_type") or last_classified, latency

        question = _extract_clarification_question(state)
        if not question:
            LOGGER.info(
                "round=%s terminated without retrieved_norms; deal_type=%r",
                round_idx,
                state.get("deal_type"),
            )
            return [], last_classified, None

        if round_idx >= max_clarifications:
            LOGGER.info("exhausted max_clarifications=%s with question still open", max_clarifications)
            return [], last_classified, None

        answer = _simulate_user_answer(simulator, initial_request, question)
        LOGGER.info("round=%s clarification question=%r → answer=%r", round_idx, question, answer)
        description = f"{description}\n\nДополнение пользователя: {answer}"

    return [], last_classified, None


_RANGE_RE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
_SOURCE_PUNCT_RE = re.compile(r"[\(\)\[\]\{\}\.,:;\"'«»]")
_SOURCE_SPACE_RE = re.compile(r"\s+")


def _parse_article_value(value: str) -> int | None:
    if value is None:
        return None
    stripped = str(value).replace(".", "").strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _normalize_source(name: str) -> str:
    value = name.replace("_", " ")
    value = _SOURCE_PUNCT_RE.sub(" ", value)
    value = value.lower()
    return _SOURCE_SPACE_RE.sub(" ", value).strip()


def _source_matches_gold(gold_clean: str, returned_clean: str) -> bool:
    if not gold_clean or not returned_clean:
        return False
    if gold_clean == returned_clean:
        return True
    return gold_clean in returned_clean or returned_clean in gold_clean


def _normalize_deal_type(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip()
    return cleaned


def _format_required_sources(gold_sources: list[dict]) -> str:
    parts: list[str] = []
    for spec in gold_sources or []:
        source = (spec.get("source_name") or "").strip()
        if not source:
            continue
        for article in spec.get("articles") or []:
            value = str(article).strip()
            if not value:
                continue
            parts.append(f"{source} ст.{value}")
    return " | ".join(parts)


def _format_actual_sources(returned: list[dict]) -> str:
    parts: list[str] = []
    for chunk in returned or []:
        source = (chunk.get("source") or "").strip() or "-"
        article = chunk.get("article_number") or chunk.get("article") or "-"
        parts.append(f"{source} ст.{str(article).strip()}")
    return " | ".join(parts)


def _classification_correct(gold: str, classified: str | None) -> bool:
    if not classified:
        return False
    return _normalize_deal_type(gold) == _normalize_deal_type(classified)


def _parse_article_spec(article_spec: str) -> tuple | None:
    article_spec = (article_spec or "").strip()
    if not article_spec:
        return None
    range_match = _RANGE_RE.match(article_spec)
    if range_match:
        lo = _parse_article_value(range_match.group(1))
        hi = _parse_article_value(range_match.group(2))
        if lo is None or hi is None or lo > hi:
            return None
        return ("range", lo, hi)
    target = _parse_article_value(article_spec)
    if target is None:
        return None
    return ("single", target)


def _flatten_gold(gold_sources: list[dict]) -> list[tuple[str, tuple]]:
    flat: list[tuple[str, tuple]] = []
    for spec in gold_sources or []:
        source_name = (spec.get("source_name") or "").strip()
        if not source_name:
            continue
        source_clean = _normalize_source(source_name)
        for article_spec in spec.get("articles") or []:
            parsed = _parse_article_spec(str(article_spec))
            if parsed is None:
                continue
            flat.append((source_clean, parsed))
    return flat


def _extract_chunk_article_value(chunk: dict) -> int | None:
    raw = chunk.get("article_number") or chunk.get("article")
    if not raw:
        return None
    value = _parse_article_value(str(raw))
    if value is not None:
        return value
    match = re.search(r"(\d+(?:\.\d+)?)", str(raw))
    if not match:
        return None
    return _parse_article_value(match.group(1))


def _spec_matches_value(parsed_spec: tuple, value: int) -> bool:
    kind = parsed_spec[0]
    if kind == "range":
        _, lo, hi = parsed_spec
        return lo <= value <= hi
    if kind == "single":
        return value == parsed_spec[1]
    return False


def _score_entry_at_k(
    returned: list[dict],
    gold_sources: list[dict],
    k_values: list[int],
) -> dict:
    flat_gold = _flatten_gold(gold_sources)
    total = len(flat_gold)

    chunk_relevance: list[bool] = [False] * len(returned)
    gold_first_match: list[int | None] = [None] * total

    for chunk_idx, chunk in enumerate(returned):
        source_raw = (chunk.get("source") or "").strip()
        if not source_raw:
            continue
        chunk_source_clean = _normalize_source(source_raw)
        chunk_value = _extract_chunk_article_value(chunk)
        if chunk_value is None:
            continue
        for gold_idx, (gold_source_clean, parsed_spec) in enumerate(flat_gold):
            if not _source_matches_gold(gold_source_clean, chunk_source_clean):
                continue
            if not _spec_matches_value(parsed_spec, chunk_value):
                continue
            chunk_relevance[chunk_idx] = True
            if gold_first_match[gold_idx] is None:
                gold_first_match[gold_idx] = chunk_idx + 1

    metrics: dict = {
        "total_gold": total,
        "matched_total": sum(1 for p in gold_first_match if p is not None),
    }
    for k in k_values:
        if k <= 0:
            continue
        if total > 0:
            matched_in_k = sum(1 for p in gold_first_match if p is not None and p <= k)
            recall = matched_in_k / total
        else:
            recall = 0.0
        relevant_in_k = sum(1 for i in range(min(k, len(chunk_relevance))) if chunk_relevance[i])
        precision = relevant_in_k / k
        hit = 1 if relevant_in_k > 0 else 0
        metrics[f"precision@{k}"] = round(precision, 4)
        metrics[f"recall@{k}"] = round(recall, 4)
        metrics[f"hit@{k}"] = hit
    return metrics


def _sanitize_for_filename(value: str) -> str:
    return _FILENAME_SAFE_RE.sub("_", value).strip("_") or "x"


def _build_csv_filename() -> str:
    parts = [_sanitize_for_filename(settings.embed_model)]
    if settings.reranker_enabled:
        parts.append(_sanitize_for_filename(settings.reranker_model))
    if settings.retrieval_filter_enabled:
        parts.append("retrievalFilter")
    parts.append(str(int(time.time())))
    return "_".join(parts) + ".csv"


def _default_output_dir() -> Path:
    return BASE_DIR / "data" / "eval_outputs"


def _default_input() -> Path:
    return BASE_DIR / "data" / "gold_sources.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Оценка качества поиска норм на эталонном датасете")
    parser.add_argument("--input", type=Path, default=None, help="gold_sources.json (default: data/gold_sources.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: data/eval_outputs")
    parser.add_argument(
        "--max-clarifications",
        type=int,
        default=3,
        help="Максимум раундов симулируемых уточнений на запись (default: 3)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="3,5,10,20",
        help="Значения k для precision@k / recall@k / hit@k через запятую (default: 3,5,10,20)",
    )
    args = parser.parse_args()

    try:
        k_values = sorted({int(v.strip()) for v in args.k_values.split(",") if v.strip() and int(v.strip()) > 0})
    except ValueError:
        parser.error(f"Invalid --k-values {args.k_values!r}; expected comma-separated positive integers")
    if not k_values:
        parser.error("--k-values produced no valid k entries")

    input_path = args.input or _default_input()
    output_dir = args.output_dir or _default_output_dir()

    if not input_path.exists():
        parser.error(f"Gold-set file not found: {input_path}")

    gold = _load_gold_set(input_path)
    if not gold:
        parser.error(f"Gold-set is empty: {input_path}")

    LOGGER.info(
        "Eval start entries=%s embed=%s reranker=%s filter=%s",
        len(gold),
        settings.embed_model,
        f"{settings.reranker_model} (on)" if settings.reranker_enabled else "off",
        "on" if settings.retrieval_filter_enabled else "off",
    )

    app = _compile_eval_graph()
    simulator = _make_user_simulator()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / _build_csv_filename()
    LOGGER.info("Writing results to %s", csv_path)

    metric_columns: list[str] = []
    for k in k_values:
        metric_columns.extend([f"precision@{k}", f"recall@{k}", f"hit@{k}"])
    header = [
        "initial_request",
        "contract_type",
        "classified_deal_type",
        "classification_correct",
        "retrieve_norms_latency_ms",
        "total_gold",
        "matched_total",
        *metric_columns,
        "required_sources",
        "actual_sources",
    ]

    classification_total = 0
    classification_correct = 0

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for idx, entry in enumerate(gold, 1):
            initial_request = (entry.get("initial_request") or "").strip()
            contract_type = (entry.get("contract_type") or "").strip()
            gold_sources = entry.get("sources") or []
            if not initial_request:
                LOGGER.warning("Skipping entry #%s with empty initial_request", idx)
                continue

            LOGGER.info("Entry %s/%s: contract_type=%r request=%r", idx, len(gold), contract_type, initial_request[:80])
            try:
                returned, classified, latency_ms = _run_agent_until_retrieve(
                    app, simulator, initial_request, args.max_clarifications
                )
            except Exception:
                LOGGER.exception("Entry %s failed; writing zeros for this entry", idx)
                returned, classified, latency_ms = [], None, None

            metrics = _score_entry_at_k(returned, gold_sources, k_values)
            is_correct = _classification_correct(contract_type, classified)
            classification_total += 1
            if is_correct:
                classification_correct += 1
            LOGGER.info(
                "Entry %s: classified=%r (gold=%r → %s) | matched=%s/%s returned=%s latency=%s ms | %s",
                idx,
                classified,
                contract_type,
                "OK" if is_correct else "MISS",
                metrics["matched_total"],
                metrics["total_gold"],
                len(returned),
                f"{latency_ms:.1f}" if latency_ms is not None else "n/a",
                " ".join(
                    f"P@{k}={metrics[f'precision@{k}']:.2f} R@{k}={metrics[f'recall@{k}']:.2f} H@{k}={metrics[f'hit@{k}']}"
                    for k in k_values
                ),
            )
            row = [
                initial_request,
                contract_type,
                classified or "",
                1 if is_correct else 0,
                round(latency_ms, 1) if latency_ms is not None else "",
                metrics["total_gold"],
                metrics["matched_total"],
            ]
            for k in k_values:
                row.extend([
                    metrics[f"precision@{k}"],
                    metrics[f"recall@{k}"],
                    metrics[f"hit@{k}"],
                ])
            row.append(_format_required_sources(gold_sources))
            row.append(_format_actual_sources(returned))
            writer.writerow(row)
            fh.flush()

    accuracy = (classification_correct / classification_total) if classification_total else 0.0
    LOGGER.info(
        "Classification accuracy: %s/%s = %.2f%%",
        classification_correct,
        classification_total,
        accuracy * 100,
    )
    LOGGER.info("Done. CSV: %s", csv_path)
    print(f"CSV: {csv_path}")
    print(f"Classification accuracy: {classification_correct}/{classification_total} = {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
