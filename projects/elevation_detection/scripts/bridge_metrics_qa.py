"""
Build a small evidence-aware QA store from bridge pier metric OCR results.

The store is intentionally deterministic: engineering quantities such as total
length, span count, pier height, and embed depth come from structured OCR output,
while every answer keeps the OCR source text and box as evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_RESULT_JSON = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "page_0019_pier_metrics_final_run"
    / "page_0019_pier_metrics.json"
)
DEFAULT_STORE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "page_0019_pier_metrics_final_run"
    / "bridge_metrics_qa_store.json"
)


def round3(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def drawing_units_to_meters(value: float | int | None) -> float | None:
    """The current drawings use centimeter-like dimensions: 2000 -> 20.0 m."""
    if value is None:
        return None
    return round(float(value) / 100.0, 3)


def source_ref(item: dict[str, Any] | None, source_type: str) -> dict[str, Any] | None:
    if not item:
        return None
    return {
        "source_type": source_type,
        "original_text": item.get("text") or item.get("token") or "",
        "token": item.get("token"),
        "value": item.get("value"),
        "confidence": item.get("confidence"),
        "bbox": item.get("bbox"),
        "center": item.get("center"),
        "source": item.get("source") or item.get("source_image"),
        "rule": item.get("source_rule"),
        "quality": item.get("candidate_quality"),
    }


def compact_none(value: Any) -> Any:
    return value if value not in ("", [], {}) else None


def build_store(result_json: Path, output: Path) -> dict[str, Any]:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    span_groups = data.get("span_groups") or []
    span_count = sum(int(group.get("span_count") or 0) for group in span_groups)
    span_length_units = [group.get("span_length") for group in span_groups if group.get("span_length")]
    total_length = data.get("total_length") or {}

    facts: list[dict[str, Any]] = []

    if total_length:
        facts.append(
            {
                "id": "bridge.total_length",
                "type": "total_length",
                "label": "桥梁总长",
                "value": total_length.get("value"),
                "unit": total_length.get("unit", "drawing_units"),
                "value_m": drawing_units_to_meters(total_length.get("value")),
                "evidence": [source_ref(total_length, "global_ocr_total_length")],
            }
        )

    if span_groups:
        facts.append(
            {
                "id": "bridge.span_count",
                "type": "span_count",
                "label": "总跨数",
                "value": span_count,
                "unit": "span",
                "evidence": [
                    source_ref(group, "global_ocr_span_group") | {
                        "span_group_index": group.get("span_group_index"),
                        "span_count": group.get("span_count"),
                        "span_length": group.get("span_length"),
                        "pier_indices": group.get("pier_indices"),
                    }
                    for group in span_groups
                    if source_ref(group, "global_ocr_span_group")
                ],
            }
        )
        for group in span_groups:
            facts.append(
                {
                    "id": f"bridge.span_group.{group.get('span_group_index')}",
                    "type": "span_group",
                    "label": f"第{group.get('span_group_index')}联跨径",
                    "span_group_index": group.get("span_group_index"),
                    "span_count": group.get("span_count"),
                    "span_length": group.get("span_length"),
                    "span_length_m": drawing_units_to_meters(group.get("span_length")),
                    "pier_indices": group.get("pier_indices"),
                    "evidence": [source_ref(group, "global_ocr_span_group")],
                }
            )

    lowest_water_level = data.get("lowest_water_level")
    if lowest_water_level:
        facts.append(
            {
                "id": "bridge.lowest_water_level",
                "type": "lowest_water_level",
                "label": "最低水位线",
                "value": lowest_water_level.get("value"),
                "unit": "m",
                "evidence": [source_ref(lowest_water_level, "global_ocr_lowest_water_level")],
            }
        )

    for item in data.get("results", []):
        pier_index = item.get("pier_index")
        selected = item.get("selected_elevations") or {}
        top = selected.get("top")
        middle = selected.get("middle")
        bottom = selected.get("bottom")
        facts.append(
            {
                "id": f"pier.{pier_index}.metrics",
                "type": "pier_metrics",
                "label": f"{pier_index}号墩指标",
                "pier_index": pier_index,
                "pier_number": (item.get("pier_number") or {}).get("number"),
                "top_elevation": compact_none(top.get("value") if top else None),
                "middle_elevation": compact_none(middle.get("value") if middle else None),
                "bottom_elevation": compact_none(bottom.get("value") if bottom else None),
                "pier_height": compact_none(item.get("pier_height")),
                "embed_depth": compact_none(item.get("embed_depth")),
                "span_group": item.get("span_group"),
                "span_length": item.get("span_length"),
                "span_length_m": drawing_units_to_meters(item.get("span_length")),
                "status": item.get("status"),
                "evidence": [
                    ref
                    for ref in [
                        source_ref(top, "selected_top_elevation"),
                        source_ref(middle, "selected_middle_elevation"),
                        source_ref(bottom, "selected_bottom_elevation"),
                    ]
                    if ref
                ],
                "local_crop": item.get("local_crop"),
            }
        )

    raw_ocr_docs = []
    for idx, row in enumerate(data.get("global_ocr_rows") or [], start=1):
        text = row.get("text", "")
        if not text:
            continue
        raw_ocr_docs.append(
            {
                "id": f"global_ocr.{idx}",
                "text": text,
                "confidence": row.get("confidence"),
                "bbox": row.get("bbox"),
                "center": row.get("center"),
                "source": row.get("source_image"),
            }
        )

    store = {
        "schema_version": 1,
        "source_result_json": str(result_json),
        "image": data.get("image"),
        "bridge_name": Path(data.get("image", "bridge")).stem,
        "summary": {
            "pier_count": data.get("pier_count"),
            "span_count": span_count or None,
            "span_lengths": sorted(set(span_length_units)),
            "total_length": total_length.get("value"),
            "total_length_m": drawing_units_to_meters(total_length.get("value")),
        },
        "facts": facts,
        "raw_ocr_docs": raw_ocr_docs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return store


def evidence_original_text(evidence: list[dict[str, Any]]) -> list[str]:
    texts = []
    for item in evidence:
        text = item.get("original_text") or item.get("text")
        if text and text not in texts:
            texts.append(text)
    return texts


def make_answer(question: str, answer: str, evidence: list[dict[str, Any]], intent: str) -> dict[str, Any]:
    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "evidence_chain": evidence,
        "original_text": evidence_original_text(evidence),
    }


def find_fact(store: dict[str, Any], fact_type: str) -> dict[str, Any] | None:
    return next((fact for fact in store["facts"] if fact.get("type") == fact_type), None)


def pier_number_from_question(question: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*号?墩", question)
    if match:
        return int(match.group(1))
    match = re.search(r"墩\s*(\d{1,2})", question)
    if match:
        return int(match.group(1))
    return None


def all_pier_metric_facts(store: dict[str, Any]) -> list[dict[str, Any]]:
    return [fact for fact in store["facts"] if fact.get("type") == "pier_metrics"]


def span_group_facts(store: dict[str, Any]) -> list[dict[str, Any]]:
    return [fact for fact in store["facts"] if fact.get("type") == "span_group"]


def answer_bearing_distance(store: dict[str, Any], question: str) -> dict[str, Any]:
    groups = span_group_facts(store)
    if groups:
        lengths = sorted({group.get("span_length") for group in groups if group.get("span_length")})
        evidence = []
        for group in groups:
            evidence.extend(group.get("evidence", []))
        if len(lengths) == 1:
            answer = (
                f"按已解析跨径推断，支座中心距/标准跨径为 "
                f"{drawing_units_to_meters(lengths[0])} m（原图标注 {lengths[0]}）。"
            )
        else:
            answer = "按已解析跨径推断，各联支座中心距/跨径为：" + "，".join(
                f"{drawing_units_to_meters(length)} m（原图标注 {length}）" for length in lengths
            ) + "。"
        return make_answer(question, answer, evidence, "bearing_distance_from_span")

    evidence = search_raw_ocr(store, ["支座", "座", "距离", "间距"])
    if evidence:
        return make_answer(question, "结构化结果中暂未形成支座距离字段，但召回到相关 OCR 原文，请人工复核。", evidence, "bearing_distance")
    return make_answer(question, "当前识别结果中没有可靠的支座距离结构化字段，也没有召回到包含“支座/距离/间距”的 OCR 原文。", [], "bearing_distance")


def answer_pier_metrics(store: dict[str, Any], question: str, pier_index: int | None) -> dict[str, Any]:
    facts = all_pier_metric_facts(store)
    if pier_index is not None:
        fact = next((item for item in facts if item.get("pier_index") == pier_index), None)
        if not fact:
            return make_answer(question, f"没有找到{pier_index}号墩的结构化识别结果。", [], "pier_metrics")
        parts = [f"{pier_index}号墩"]
        if "墩高" in question or "高度" in question or "高" in question:
            parts.append(f"墩高={fact.get('pier_height')} m" if fact.get("pier_height") is not None else "墩高未能可靠计算")
        if "埋深" in question or "深" in question:
            parts.append(f"埋深={fact.get('embed_depth')} m" if fact.get("embed_depth") is not None else "埋深未能可靠计算")
        if len(parts) == 1:
            parts.extend(
                [
                    f"墩高={fact.get('pier_height')} m" if fact.get("pier_height") is not None else "墩高未能可靠计算",
                    f"埋深={fact.get('embed_depth')} m" if fact.get("embed_depth") is not None else "埋深未能可靠计算",
                ]
            )
        return make_answer(question, "，".join(parts) + "。", fact.get("evidence", []), "pier_metrics")

    lines = []
    evidence = []
    for fact in facts:
        lines.append(
            f"{fact['pier_index']}号墩: 墩高={fact.get('pier_height')}, 埋深={fact.get('embed_depth')}, "
            f"top={fact.get('top_elevation')}, middle={fact.get('middle_elevation')}, bottom={fact.get('bottom_elevation')}"
        )
        evidence.extend(fact.get("evidence", [])[:1])
    return make_answer(question, "\n".join(lines), evidence[:20], "pier_metrics_all")


def search_raw_ocr(store: dict[str, Any], keywords: list[str]) -> list[dict[str, Any]]:
    hits = []
    for row in store.get("raw_ocr_docs", []):
        text = row.get("text", "")
        if any(keyword in text for keyword in keywords):
            hits.append(
                {
                    "source_type": "global_ocr_raw_text",
                    "original_text": text,
                    "confidence": row.get("confidence"),
                    "bbox": row.get("bbox"),
                    "center": row.get("center"),
                    "source": row.get("source"),
                }
            )
    return hits


def ask(store: dict[str, Any], question: str) -> dict[str, Any]:
    normalized = question.replace(" ", "")

    if any(word in normalized for word in ["总跨多少米", "全长", "总长", "桥长", "长度"]):
        fact = find_fact(store, "total_length")
        if fact:
            answer = f"识别到桥梁总长为 {fact.get('value_m')} m（原图标注值 {fact.get('value')}）。"
            return make_answer(question, answer, fact.get("evidence", []), "total_length")

    if any(word in normalized for word in ["多少跨", "几跨", "跨数"]):
        fact = find_fact(store, "span_count")
        if fact:
            group_desc = []
            for ev in fact.get("evidence", []):
                group_desc.append(
                    f"第{ev.get('span_group_index')}联 {ev.get('span_count')}跨 x {drawing_units_to_meters(ev.get('span_length'))}m"
                )
            answer = f"识别到总跨数为 {fact.get('value')} 跨" + ("；" + "，".join(group_desc) if group_desc else "") + "。"
            return make_answer(question, answer, fact.get("evidence", []), "span_count")

    if any(word in normalized for word in ["支座距离", "支座间距", "支座"]):
        return answer_bearing_distance(store, question)

    if any(word in normalized for word in ["墩高", "埋深", "高程", "墩"]):
        return answer_pier_metrics(store, question, pier_number_from_question(normalized))

    evidence = search_raw_ocr(store, list(normalized[:8]))
    return make_answer(question, "暂未命中结构化问答规则，可查看召回 OCR 原文后扩展解析规则。", evidence[:10], "unknown")


def _cn(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


CN_TOTAL_TERMS = [_cn(r"\u603b\u8de8\u591a\u5c11\u7c73"), _cn(r"\u5168\u957f"), _cn(r"\u603b\u957f"), _cn(r"\u6865\u957f"), _cn(r"\u957f\u5ea6")]
CN_SPAN_TERMS = [_cn(r"\u591a\u5c11\u8de8"), _cn(r"\u51e0\u8de8"), _cn(r"\u8de8\u6570")]
CN_BEARING_TERMS = [_cn(r"\u652f\u5ea7\u8ddd\u79bb"), _cn(r"\u652f\u5ea7\u95f4\u8ddd"), _cn(r"\u652f\u5ea7")]
CN_PIER_TERMS = [_cn(r"\u58a9\u9ad8"), _cn(r"\u57cb\u6df1"), _cn(r"\u9ad8\u7a0b"), _cn(r"\u58a9")]
CN_HIGHEST_TERMS = [_cn(r"\u6700\u9ad8"), _cn(r"\u6700\u5927\u58a9\u9ad8"), _cn(r"\u54ea\u4e2a\u58a9\u6700\u9ad8")]
CN_DEEPEST_TERMS = [_cn(r"\u6700\u6df1"), _cn(r"\u6700\u5927\u57cb\u6df1"), _cn(r"\u54ea\u4e2a\u58a9\u57cb\u6df1\u6700\u5927")]


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _fmt_m(value: Any) -> str:
    if value is None:
        return _cn(r"\u672a\u80fd\u53ef\u9760\u8ba1\u7b97")
    return f"{value} m"


def _robust_pier_number(question: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*(?:\u53f7)?\s*\u58a9", question)
    if match:
        return int(match.group(1))
    match = re.search(r"\u58a9\s*(\d{1,2})", question)
    if match:
        return int(match.group(1))
    return None


def _span_group_number(question: str) -> int | None:
    match = re.search(r"\u7b2c?\s*(\d{1,2})\s*\u8054", question)
    if match:
        return int(match.group(1))
    return None


def _pier_fact(store: dict[str, Any], pier_index: int) -> dict[str, Any] | None:
    return next(
        (
            fact
            for fact in all_pier_metric_facts(store)
            if int(fact.get("pier_index") or -1) == int(pier_index)
        ),
        None,
    )


def _answer_total_length(store: dict[str, Any], question: str) -> dict[str, Any]:
    fact = find_fact(store, "total_length")
    if not fact:
        return make_answer(question, _cn(r"\u6ca1\u6709\u627e\u5230\u6865\u6881\u603b\u957f\u7684\u7ed3\u6784\u5316\u8bc6\u522b\u7ed3\u679c\u3002"), [], "total_length")
    answer = _cn(r"\u8bc6\u522b\u5230\u6865\u6881\u603b\u957f\u4e3a ") + f"{fact.get('value_m')} m" + _cn(r"\uff08\u539f\u56fe\u6807\u6ce8\u503c ") + f"{fact.get('value')}" + _cn(r"\uff09\u3002")
    return make_answer(question, answer, fact.get("evidence", []), "total_length")


def _answer_span_count(store: dict[str, Any], question: str) -> dict[str, Any]:
    fact = find_fact(store, "span_count")
    if not fact:
        return make_answer(question, _cn(r"\u6ca1\u6709\u627e\u5230\u8de8\u6570\u7684\u7ed3\u6784\u5316\u8bc6\u522b\u7ed3\u679c\u3002"), [], "span_count")
    group_desc = []
    for ev in fact.get("evidence", []):
        group_desc.append(
            _cn(r"\u7b2c") + f"{ev.get('span_group_index')}" + _cn(r"\u8054 ")
            + f"{ev.get('span_count')}" + _cn(r"\u8de8 x ")
            + f"{drawing_units_to_meters(ev.get('span_length'))}m"
        )
    answer = _cn(r"\u8bc6\u522b\u5230\u603b\u8de8\u6570\u4e3a ") + f"{fact.get('value')}" + _cn(r" \u8de8")
    if group_desc:
        answer += _cn(r"\uff1b") + _cn(r"\uff0c").join(group_desc)
    answer += _cn(r"\u3002")
    return make_answer(question, answer, fact.get("evidence", []), "span_count")


def _answer_span_group(store: dict[str, Any], question: str, group_index: int) -> dict[str, Any]:
    group = next((fact for fact in span_group_facts(store) if int(fact.get("span_group_index") or -1) == group_index), None)
    if not group:
        return make_answer(question, _cn(r"\u6ca1\u6709\u627e\u5230\u7b2c") + f"{group_index}" + _cn(r"\u8054\u7684\u8de8\u5f84\u4fe1\u606f\u3002"), [], "span_group")
    answer = (
        _cn(r"\u7b2c") + f"{group_index}" + _cn(r"\u8054\u4e3a ")
        + f"{group.get('span_count')}" + _cn(r"\u8de8 x ")
        + f"{group.get('span_length_m')} m"
        + _cn(r"\uff08\u539f\u56fe\u6807\u6ce8 ")
        + f"{group.get('span_length')}" + _cn(r"\uff09\u3002")
    )
    return make_answer(question, answer, group.get("evidence", []), "span_group")


def _answer_bearing_distance(store: dict[str, Any], question: str) -> dict[str, Any]:
    groups = span_group_facts(store)
    if groups:
        lengths = sorted({group.get("span_length") for group in groups if group.get("span_length")})
        evidence: list[dict[str, Any]] = []
        for group in groups:
            evidence.extend(group.get("evidence", []))
        if len(lengths) == 1:
            answer = (
                _cn(r"\u6309\u5df2\u89e3\u6790\u8de8\u5f84\u63a8\u65ad\uff0c\u652f\u5ea7\u4e2d\u5fc3\u8ddd/\u6807\u51c6\u8de8\u5f84\u4e3a ")
                + f"{drawing_units_to_meters(lengths[0])} m"
                + _cn(r"\uff08\u539f\u56fe\u6807\u6ce8 ")
                + f"{lengths[0]}" + _cn(r"\uff09\u3002")
            )
        else:
            answer = _cn(r"\u6309\u5df2\u89e3\u6790\u8de8\u5f84\u63a8\u65ad\uff0c\u5404\u8054\u652f\u5ea7\u4e2d\u5fc3\u8ddd/\u8de8\u5f84\u4e3a\uff1a")
            answer += _cn(r"\uff0c").join(f"{drawing_units_to_meters(length)} m" for length in lengths) + _cn(r"\u3002")
        return make_answer(question, answer, evidence, "bearing_distance_from_span")
    evidence = search_raw_ocr(store, [_cn(r"\u652f\u5ea7"), _cn(r"\u8ddd\u79bb"), _cn(r"\u95f4\u8ddd")])
    return make_answer(question, _cn(r"\u5f53\u524d\u6ca1\u6709\u53ef\u9760\u7684\u652f\u5ea7\u8ddd\u79bb\u7ed3\u6784\u5316\u5b57\u6bb5\u3002"), evidence, "bearing_distance")


def _answer_pier_metrics(store: dict[str, Any], question: str, pier_index: int | None) -> dict[str, Any]:
    facts = all_pier_metric_facts(store)
    if pier_index is not None:
        fact = _pier_fact(store, pier_index)
        if not fact:
            return make_answer(question, _cn(r"\u6ca1\u6709\u627e\u5230") + f"{pier_index}" + _cn(r"\u53f7\u58a9\u7684\u7ed3\u6784\u5316\u8bc6\u522b\u7ed3\u679c\u3002"), [], "pier_metrics")
        wants_height = _cn(r"\u58a9\u9ad8") in question or _cn(r"\u9ad8\u5ea6") in question
        wants_depth = _cn(r"\u57cb\u6df1") in question or _cn(r"\u6df1") in question
        wants_elev = _cn(r"\u9ad8\u7a0b") in question or _cn(r"\u6807\u9ad8") in question
        parts = [_cn(r"\u7b2c") + f"{pier_index}" + _cn(r"\u53f7\u58a9")]
        if wants_elev:
            parts.append(
                f"top={_fmt_m(fact.get('top_elevation'))}, "
                f"middle={_fmt_m(fact.get('middle_elevation'))}, "
                f"bottom={_fmt_m(fact.get('bottom_elevation'))}"
            )
        if wants_height or not (wants_depth or wants_elev):
            parts.append(_cn(r"\u58a9\u9ad8=") + _fmt_m(fact.get("pier_height")))
        if wants_depth or not (wants_height or wants_elev):
            parts.append(_cn(r"\u57cb\u6df1=") + _fmt_m(fact.get("embed_depth")))
        return make_answer(question, _cn(r"\uff0c").join(parts) + _cn(r"\u3002"), fact.get("evidence", []), "pier_metrics")

    lines = []
    evidence: list[dict[str, Any]] = []
    for fact in facts:
        lines.append(
            f"{fact['pier_index']}" + _cn(r"\u53f7\u58a9: \u58a9\u9ad8=")
            + f"{fact.get('pier_height')}, " + _cn(r"\u57cb\u6df1=")
            + f"{fact.get('embed_depth')}, top={fact.get('top_elevation')}, "
            + f"middle={fact.get('middle_elevation')}, bottom={fact.get('bottom_elevation')}"
        )
        evidence.extend(fact.get("evidence", [])[:1])
    return make_answer(question, "\n".join(lines), evidence[:20], "pier_metrics_all")


def _answer_ranked_pier(store: dict[str, Any], question: str, metric: str) -> dict[str, Any]:
    facts = [fact for fact in all_pier_metric_facts(store) if fact.get(metric) is not None]
    if not facts:
        label = _cn(r"\u58a9\u9ad8") if metric == "pier_height" else _cn(r"\u57cb\u6df1")
        return make_answer(question, _cn(r"\u6ca1\u6709\u53ef\u7528\u7684") + label + _cn(r"\u6570\u636e\u3002"), [], f"rank_{metric}")
    best = max(facts, key=lambda fact: float(fact.get(metric) or 0))
    label = _cn(r"\u58a9\u9ad8") if metric == "pier_height" else _cn(r"\u57cb\u6df1")
    answer = (
        _cn(r"\u5df2\u8bc6\u522b\u7ed3\u679c\u4e2d\uff0c")
        + f"{best.get('pier_index')}" + _cn(r"\u53f7\u58a9\u7684")
        + label + _cn(r"\u6700\u5927\uff0c\u4e3a ")
        + f"{best.get(metric)} m" + _cn(r"\u3002")
    )
    return make_answer(question, answer, best.get("evidence", []), f"rank_{metric}")


def retrieve_for_question(store: dict[str, Any], question: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", "", question)
    group_index = _span_group_number(normalized)
    pier_index = _robust_pier_number(normalized)
    if _has_any(normalized, CN_HIGHEST_TERMS):
        return _answer_ranked_pier(store, question, "pier_height")
    if _has_any(normalized, CN_DEEPEST_TERMS):
        return _answer_ranked_pier(store, question, "embed_depth")
    if group_index and (_cn(r"\u8de8") in normalized or _cn(r"\u8054") in normalized):
        return _answer_span_group(store, question, group_index)
    if _has_any(normalized, CN_TOTAL_TERMS) or (_cn(r"\u591a\u5c11\u7c73") in normalized and _cn(r"\u8de8") in normalized):
        return _answer_total_length(store, question)
    if _has_any(normalized, CN_SPAN_TERMS):
        return _answer_span_count(store, question)
    if _has_any(normalized, CN_BEARING_TERMS):
        return _answer_bearing_distance(store, question)
    if pier_index is not None or _has_any(normalized, CN_PIER_TERMS):
        return _answer_pier_metrics(store, question, pier_index)
    evidence = search_raw_ocr(store, [ch for ch in normalized[:8] if ch.strip()])
    return make_answer(question, _cn(r"\u6682\u672a\u547d\u4e2d\u7ed3\u6784\u5316\u95ee\u7b54\u89c4\u5219\uff0c\u53ef\u67e5\u770b\u53ec\u56de OCR \u539f\u6587\u540e\u6269\u5c55\u89e3\u6790\u89c4\u5219\u3002"), evidence[:10], "unknown")


def _llm_configured() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("BRIDGE_QA_API_KEY")
        or os.environ.get("MIMO_API_KEY")
        or os.environ.get("MIMO_TOKEN")
    )


def _openai_compatible_chat(question: str, retrieval: dict[str, Any]) -> str | None:
    if not _llm_configured():
        return None
    try:
        import os
        import requests
    except Exception:
        return None
    api_key = (
        os.environ.get("BRIDGE_QA_API_KEY")
        or os.environ.get("MIMO_API_KEY")
        or os.environ.get("MIMO_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
    )
    base_url = (
        os.environ.get("BRIDGE_QA_BASE_URL")
        or os.environ.get("MIMO_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = os.environ.get("BRIDGE_QA_MODEL") or os.environ.get("MIMO_MODEL") or "gpt-4o-mini"
    context = {
        "question": question,
        "structured_answer": retrieval.get("answer"),
        "intent": retrieval.get("intent"),
        "evidence_chain": retrieval.get("evidence_chain", [])[:12],
        "original_text": retrieval.get("original_text", [])[:12],
    }
    system = (
        "You answer Chinese bridge drawing QA. Use only the provided structured data "
        "and evidence. Do not invent numbers. If data is missing, say it is missing. "
        "Mention key evidence briefly."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "temperature": 0.1,
    }
    try:
        resp = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def ask(store: dict[str, Any], question: str) -> dict[str, Any]:
    retrieval = retrieve_for_question(store, question)
    llm_answer = _openai_compatible_chat(question, retrieval)
    if llm_answer:
        enhanced = dict(retrieval)
        enhanced["answer"] = llm_answer
        enhanced["intent"] = f"llm_{retrieval.get('intent', 'unknown')}"
        enhanced["llm_enhanced"] = True
        return enhanced
    retrieval["llm_enhanced"] = False
    return retrieval


def print_markdown(result: dict[str, Any]) -> None:
    print(f"问题：{result['question']}")
    print(f"答案：{result['answer']}")
    print("证据链：")
    if not result["evidence_chain"]:
        print("- 无")
    for item in result["evidence_chain"]:
        print(
            f"- 来源={item.get('source_type')} 原文={item.get('original_text')} "
            f"置信度={item.get('confidence')} bbox={item.get('bbox')}"
        )
    print("原文：")
    if result["original_text"]:
        for text in result["original_text"]:
            print(f"- {text}")
    else:
        print("- 无")


def print_markdown(result: dict[str, Any]) -> None:
    print(_cn(r"\u95ee\u9898\uff1a") + str(result["question"]))
    print(_cn(r"\u7b54\u6848\uff1a") + str(result["answer"]))
    print(_cn(r"\u8bc1\u636e\u94fe\uff1a"))
    if not result["evidence_chain"]:
        print("- " + _cn(r"\u65e0"))
    for item in result["evidence_chain"]:
        print(
            "- "
            + _cn(r"\u6765\u6e90=")
            + str(item.get("source_type"))
            + " "
            + _cn(r"\u539f\u6587=")
            + str(item.get("original_text"))
            + " "
            + _cn(r"\u7f6e\u4fe1\u5ea6=")
            + str(item.get("confidence"))
            + " bbox="
            + str(item.get("bbox"))
        )
    print(_cn(r"\u539f\u6587\uff1a"))
    if result["original_text"]:
        for text in result["original_text"]:
            print(f"- {text}")
    else:
        print("- " + _cn(r"\u65e0"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query bridge metric OCR QA store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build QA store from pier metric JSON.")
    build_parser.add_argument("--input", type=Path, default=DEFAULT_RESULT_JSON)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_STORE)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against a QA store.")
    ask_parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        store = build_store(args.input, args.output)
        print(f"Wrote QA store: {args.output}")
        print(json.dumps(store["summary"], ensure_ascii=False, indent=2))
        return

    store = json.loads(args.store.read_text(encoding="utf-8"))
    result = ask(store, args.question)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_markdown(result)


if __name__ == "__main__":
    main()
