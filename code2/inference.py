# inference.py
"""
统一推理入口：按模型名分流 Qwen / Gemma-4 调用方式。

Qwen 系列:
  - 关 thinking: chat_template_kwargs.enable_thinking=False
  - 开 thinking: 不传（默认开启）
  - 思考边界: </think>

Gemma-4 系列 (名称含 gemma-4):
  - 显式 enable_thinking True/False
  - 开 thinking 时 skip_special_tokens=False，保留
    <|channel>thought ... <channel|>
  - 思考边界: <channel|>
"""

from __future__ import annotations

import math
import re

import os

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(
    api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
)

CHOICES = ["A", "B", "C", "D", "E"]
_CHOICE_CLASS = "".join(CHOICES)

ANSWER_MARKER_RE = re.compile(r"(?:答案|answer)\s*[:：]", re.IGNORECASE)

# Qwen
THINK_END_TAG = "</" + "think>"
QWEN_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Gemma-4
CHANNEL_END = "<channel|>"
CHANNEL_START_RE = re.compile(r"<\|channel>thought\s*", re.IGNORECASE)


def is_gemma4(model: str) -> bool:
    return "gemma-4" in (model or "").lower()


def build_extra_body(model: str, enable_thinking: bool) -> dict | None:
    """按模型构造 chat.completions 的 extra_body。"""
    if is_gemma4(model):
        extra: dict = {
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        if enable_thinking:
            # 必须保留特殊 token，否则看不到 <channel|> 边界
            extra["skip_special_tokens"] = False
        return extra

    # Qwen 等：开 thinking 时走服务端默认；关 thinking 显式关闭
    if enable_thinking:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}


def split_gemma_thought_answer(text: str) -> tuple[str, str]:
    """Gemma-4: (<|channel>thought ... <channel|>)(answer)。"""
    if not text:
        return "", ""
    if CHANNEL_END not in text:
        return "", text.strip()
    thought_raw, answer = text.split(CHANNEL_END, 1)
    thought = CHANNEL_START_RE.sub("", thought_raw).strip()
    answer = answer.replace("<end_of_turn>", "").replace("<eos>", "").strip()
    return thought, answer


def strip_thinking(text: str | None) -> str:
    """去掉各模型思考块，只保留最终回答文本（写入多轮 history / 解析答案）。"""
    if not text:
        return ""
    text = QWEN_THINK_RE.sub("", text)
    if CHANNEL_END in text or CHANNEL_START_RE.search(text):
        _, answer = split_gemma_thought_answer(text)
        return answer
    return text.strip()


def ask_model(model, prompt, *, enable_thinking=False):
    create_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10000,
    }
    extra_body = build_extra_body(model, enable_thinking)
    if extra_body is not None:
        create_kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**create_kwargs)
    return (resp.choices[0].message.content or "").strip()


def _parse_letter_after_marker(tail: str) -> str | None:
    tail = tail.strip()
    if not tail:
        return None
    for pattern in (
        rf"^\s*\*{{0,2}}([{_CHOICE_CLASS}])\*{{0,2}}",
        rf"^\s*([{_CHOICE_CLASS}])\b",
        rf"([{_CHOICE_CLASS}])",
    ):
        match = re.search(pattern, tail, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def parse_answer(text):
    if text is None:
        return "UNKNOWN"

    text = strip_thinking(text)
    if not text:
        return "UNKNOWN"

    simple_text = re.sub(r"[.\s：:]", "", text).upper()
    if len(simple_text) == 1 and simple_text in CHOICES:
        return simple_text

    marker_matches = list(ANSWER_MARKER_RE.finditer(text))
    if marker_matches:
        tail = text[marker_matches[-1].end() :].strip()
        letter = _parse_letter_after_marker(tail)
        if letter:
            return letter

    return "UNKNOWN"


# ---- 完成状态 / 非终止（诱导后长循环未收尾）----
COMPLETION_COMPLETE = "complete"
COMPLETION_THINK_UNCLOSED = "think_unclosed"
COMPLETION_NO_FINAL_ANSWER = "no_final_answer"
COMPLETION_CONTEXT_OVERFLOW = "context_overflow"


def detect_think_unclosed(text: str | None) -> bool:
    """
    思考块已开启但未闭合（典型：长循环后撞上下文/长度上限被截断）。
    Qwen: <think> ... 无 </think>
    Gemma-4: <|channel>thought ... 无 <channel|>
    """
    if not text:
        return False
    has_qwen_open = "<think>" in text
    has_qwen_close = THINK_END_TAG in text
    if has_qwen_open and not has_qwen_close:
        return True
    has_gemma_open = bool(CHANNEL_START_RE.search(text)) or "<|channel>thought" in text
    has_gemma_close = CHANNEL_END in text
    if has_gemma_open and not has_gemma_close:
        return True
    return False


def classify_completion_status(
    text: str | None,
    *,
    parsed_answer: str | None = None,
    finish_reason: str | None = None,
    error_tag: str | None = None,
) -> str:
    """
    单轮/单次生成的完成状态（忠实记录，不试图「修复」模型输出）。

    - complete: 思考已闭合（或无思考标签），且能解析出 A–E
    - think_unclosed: 思考未闭合（非终止长循环 / 截断）
    - no_final_answer: 思考已闭合但无法解析最终选项
    - context_overflow: 多轮上下文溢出（由 error_tag 标记）
    """
    if error_tag == "context_overflow":
        return COMPLETION_CONTEXT_OVERFLOW
    if detect_think_unclosed(text):
        return COMPLETION_THINK_UNCLOSED
    parsed = parsed_answer if parsed_answer is not None else parse_answer(text)
    if parsed not in CHOICES:
        return COMPLETION_NO_FINAL_ANSWER
    # finish_reason=length 但已闭合且已解析出答案：仍视为 complete
    _ = finish_reason
    return COMPLETION_COMPLETE


def softmax_from_logprobs(logprob_dict: dict[str, float]) -> dict[str, float]:
    if not logprob_dict:
        return {}
    exp_scores = {k: math.exp(v) for k, v in logprob_dict.items()}
    total = sum(exp_scores.values())
    return {k: exp_scores[k] / total for k in exp_scores}


def _think_end_idx(content_logprobs) -> int:
    """Qwen: </think>；Gemma-4: <channel|>。取最后一个匹配位置。"""
    end = -1
    for i, item in enumerate(content_logprobs):
        tok = item.token
        if THINK_END_TAG in tok or CHANNEL_END in tok:
            end = i
    return end


def _is_special_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return True
    if t.startswith("<|") and t.endswith("|>"):
        return True
    if "redacted" in t.lower():
        return True
    if t in (CHANNEL_END, "<end_of_turn>", "<eos>"):
        return True
    return False


def _token_is_choice_token(token: str) -> bool:
    return (not _is_special_token(token)) and token.strip().upper() in CHOICES


def locate_decision_token_index(content_logprobs, full_text: str) -> tuple[int, str]:
    """
    在 token 流内定位决策点（跳过 thinking 段）。
    1. 找最后一个「答案：/ Answer:」后的首个选项字母 token
    2. fallback：从后往前最后一个 A/B/C/D/E token
    """
    think_end = _think_end_idx(content_logprobs)
    search_start = think_end + 1 if think_end >= 0 else 0

    accumulated = ""
    last_marker_end = -1
    for i in range(search_start, len(content_logprobs)):
        accumulated += content_logprobs[i].token
        matches = list(ANSWER_MARKER_RE.finditer(accumulated))
        if matches:
            last_marker_end = matches[-1].end()

    if last_marker_end >= 0:
        pos = 0
        for i in range(search_start, len(content_logprobs)):
            token_start = pos
            pos += len(content_logprobs[i].token)
            if token_start < last_marker_end:
                continue
            if _token_is_choice_token(content_logprobs[i].token):
                return i, "answer_marker"

    for i in range(len(content_logprobs) - 1, search_start - 1, -1):
        if _token_is_choice_token(content_logprobs[i].token):
            return i, "last_letter"

    for i in range(len(content_logprobs) - 1, search_start - 1, -1):
        if not _is_special_token(content_logprobs[i].token):
            return i, "fallback_last_token"

    return len(content_logprobs) - 1, "fallback_last_token"


def ask_with_belief(
    model: str,
    messages: list[dict],
    *,
    enable_thinking: bool = False,
    verbose: bool = False,
) -> dict:
    """
    带 logprob 信念探测的推理（filter / run_benchmark / run_drift 共用）。
    """
    create_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
    }
    extra_body = build_extra_body(model, enable_thinking)
    if extra_body is not None:
        create_kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**create_kwargs)
    choice0 = resp.choices[0]
    finish_reason = getattr(choice0, "finish_reason", None)
    content_logprobs = choice0.logprobs.content
    full_text = (choice0.message.content or "").strip()
    answer_text = strip_thinking(full_text)

    target_index, decision_method = locate_decision_token_index(
        content_logprobs, full_text
    )
    decision_info = content_logprobs[target_index]
    token_upper = decision_info.token.strip().upper()
    decision_valid = int(token_upper in CHOICES)

    raw_logprobs: dict[str, float] = {}
    if token_upper in CHOICES:
        raw_logprobs[token_upper] = decision_info.logprob

    for top in decision_info.top_logprobs:
        token = top.token.strip().upper()
        if token in CHOICES:
            raw_logprobs[token] = max(
                raw_logprobs.get(token, float("-inf")), top.logprob
            )

    for choice in CHOICES:
        raw_logprobs.setdefault(choice, -100.0)

    belief = softmax_from_logprobs(raw_logprobs)
    parsed = parse_answer(answer_text)
    pred = token_upper if decision_valid else parsed
    pred_matches_parse = int(pred == parsed)

    completion_status = classify_completion_status(
        full_text,
        parsed_answer=parsed,
        finish_reason=finish_reason,
    )
    is_complete = int(completion_status == COMPLETION_COMPLETE)

    # 思考未闭合时 last_letter 不可信：保留 target_token 供诊断，pred 记为 UNKNOWN
    if completion_status == COMPLETION_THINK_UNCLOSED:
        decision_valid = 0
        pred = "UNKNOWN"
        pred_matches_parse = int(pred == parsed)

    if verbose:
        print(
            f"DEBUG: model={model} gemma4={is_gemma4(model)} "
            f"token='{decision_info.token}' index={target_index} "
            f"method={decision_method} valid={bool(decision_valid)} "
            f"completion={completion_status} finish_reason={finish_reason}"
        )
        print("full_text:", full_text)
        print("answer_text:", answer_text)
        print("belief:", belief)
        print("raw_logprobs:", raw_logprobs)
        print(f"pred={pred} parse_answer={parsed} match={bool(pred_matches_parse)}")

    return {
        "messages": messages,
        "text": full_text,
        "answer_text": answer_text,
        "belief": belief,
        "raw_logprobs": raw_logprobs,
        "target_token": decision_info.token,
        "target_index": target_index,
        "decision_method": decision_method,
        "decision_valid": decision_valid,
        "pred": pred,
        "parsed_answer": parsed,
        "pred_matches_parse": pred_matches_parse,
        "finish_reason": finish_reason,
        "completion_status": completion_status,
        "is_complete": is_complete,
        "raw_len": len(full_text),
    }
