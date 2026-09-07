# backend.py
"""
Transformers 本地推理后端（替代原 inference.py 的 vLLM/OpenAI API 版）。

对齐 Qwen3 官方使用规范：
  - think 模式硬开关：apply_chat_template(enable_thinking=...)（非提示词软约束）。
  - 采样参数遵循官方 generation_config（think: 0.6/0.95/20；非 think: 0.7/0.8/20），
    不使用贪心解码（官方指出贪心在 think 模式易退化/无尽重复）。
  - 响应解析：按 </think>(token id 151668) 在 token 序列上硬拆 thinking_content 与 content，
    答案仅从 content（正式回答）中解析。
  - 信念探测：用 output_logits（原始 logits）而非 output_scores（被采样参数 warp 过），
    使 belief 反映模型真实分布、与采样温度解耦。

返回 schema（向后兼容旧字段，另增 thinking_content / content / full_text）：
  text / content / thinking_content / full_text / belief / raw_logprobs /
  target_token / target_index / decision_method / decision_valid /
  pred / parsed_answer / pred_matches_parse / messages
"""

from __future__ import annotations

import math
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from model_adapt import resolve_think_end_id

# 答案标记（"答案：" / "answer:"）
ANSWER_MARKER_RE = re.compile(r"(?:答案|answer)\s*[:：]", re.IGNORECASE)
THINK_END_TAG = "</" + "think>"


# ===================== 答案解析（作用于 content，即 think 之后的正式回答） =====================
def _parse_letter_after_marker(tail: str, choices: list[str]) -> str | None:
    tail = tail.strip()
    if not tail:
        return None
    choice_class = "".join(choices)
    for pattern in (
        rf"^\s*\*{{0,2}}([{choice_class}])\*{{0,2}}",
        rf"^\s*([{choice_class}])\b",
        rf"([{choice_class}])",
    ):
        match = re.search(pattern, tail, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def parse_answer(text: str | None, choices: list[str] | None = None) -> str:
    """从 content（正式回答）中解析最终选项字母，失败返回 'UNKNOWN'。"""
    if choices is None:
        choices = config.CHOICES_MAX
    if text is None:
        return "UNKNOWN"

    text = text.strip()
    if not text:
        return "UNKNOWN"

    simple_text = re.sub(r"[.\s：:]", "", text).upper()
    if len(simple_text) == 1 and simple_text in choices:
        return simple_text

    marker_matches = list(ANSWER_MARKER_RE.finditer(text))
    if marker_matches:
        tail = text[marker_matches[-1].end():].strip()
        letter = _parse_letter_after_marker(tail, choices)
        if letter:
            return letter

    return "UNKNOWN"


def softmax_from_logprobs(logprob_dict: dict[str, float]) -> dict[str, float]:
    if not logprob_dict:
        return {}
    exp_scores = {k: math.exp(v) for k, v in logprob_dict.items()}
    total = sum(exp_scores.values())
    if total <= 0:
        return {k: 0.0 for k in logprob_dict}
    return {k: exp_scores[k] / total for k in exp_scores}


# ===================== 决策 token 定位 =====================
def _is_special_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return True
    if t.startswith("<|") and t.endswith("|>"):
        return True
    if "redacted" in t.lower():
        return True
    return False


def _token_is_choice_token(token: str, choices: list[str]) -> bool:
    return (not _is_special_token(token)) and token.strip().upper() in choices


def locate_decision_token_index(
    token_strs: list[str], choices: list[str], search_start: int = 0
) -> tuple[int, str]:
    """
    在 content 区间（search_start 起，即 </think> 之后）定位决策 token：
    1. 优先取最后一个「答案：/ Answer:」后的首个选项字母 token；
    2. 回退：从后往前最后一个选项字母 token；
    3. 再回退：最后一个非特殊 token。
    """
    search_start = max(0, min(search_start, len(token_strs)))

    accumulated = ""
    last_marker_end = -1
    for i in range(search_start, len(token_strs)):
        accumulated += token_strs[i]
        matches = list(ANSWER_MARKER_RE.finditer(accumulated))
        if matches:
            last_marker_end = matches[-1].end()

    if last_marker_end >= 0:
        pos = 0
        for i in range(search_start, len(token_strs)):
            token_start = pos
            pos += len(token_strs[i])
            if token_start < last_marker_end:
                continue
            if _token_is_choice_token(token_strs[i], choices):
                return i, "answer_marker"

    for i in range(len(token_strs) - 1, search_start - 1, -1):
        if _token_is_choice_token(token_strs[i], choices):
            return i, "last_letter"

    for i in range(len(token_strs) - 1, search_start - 1, -1):
        if not _is_special_token(token_strs[i]):
            return i, "fallback_last_token"

    return max(search_start, len(token_strs) - 1), "fallback_last_token"


# ===================== 本地模型封装 =====================
class LocalModel:
    """封装 Transformers 模型加载、chat 模板、官方采样生成与信念探测。"""

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        *,
        dtype=config.DTYPE,
        device=config.DEVICE,
        attn_implementation: str = config.ATTN_IMPLEMENTATION,
        max_new_tokens: int | None = None,
    ) -> None:
        self.model_path = model_path
        # None 表示按 think 模式动态取 config.max_new_tokens_for()
        self.max_new_tokens_override = max_new_tokens
        # 目标设备：优先构造参数，否则用 config.DEVICE；统一经 normalize_device
        device = config.normalize_device(device if device is not None else config.DEVICE)
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            self.device = "cpu"
        else:
            self.device = device
        print(f"⏳ 正在加载模型与分词器: {model_path}  ->  device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 不使用 device_map：先加载再 .to(device)，绕开 transformers 5.x 的
        # caching_allocator_warmup 在特定 CUDA 初始化时序下报 "invalid device ordinal" 的问题。
        load_kwargs = dict(attn_implementation=attn_implementation)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype, **load_kwargs
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, **load_kwargs
            )
        self.model = self.model.to(self.device)
        self.model.eval()

        # think 结束 token：跨模型解析（Qwen </think>；Gemma 等见 model_adapt）
        end_id = resolve_think_end_id(self.tokenizer)
        if end_id is None:
            end_id = self.tokenizer.convert_tokens_to_ids(THINK_END_TAG)
        if end_id is None or end_id < 0:
            end_id = config.THINK_END_TOKEN_ID
        self.think_end_id = end_id

        # 预计算每个候选字母对应的单 token id 集合（含大小写与前导空格变体）
        self._choice_token_ids: dict[str, set[int]] = {
            letter: self._collect_choice_token_ids(letter)
            for letter in config.CHOICES_MAX
        }

    def _collect_choice_token_ids(self, letter: str) -> set[int]:
        ids: set[int] = set()
        for variant in (letter, " " + letter, letter.lower(), " " + letter.lower()):
            enc = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return ids

    def _max_new_tokens(self, enable_thinking: bool) -> int:
        if self.max_new_tokens_override is not None:
            return self.max_new_tokens_override
        return config.max_new_tokens_for(enable_thinking)

    def build_prompt(self, messages: list[dict], *, enable_thinking: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def count_prompt_tokens(self, messages: list[dict], *, enable_thinking: bool) -> int:
        prompt = self.build_prompt(messages, enable_thinking=enable_thinking)
        return len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])

    def _split_thinking(self, gen_ids: list[int]) -> tuple[str, str, int]:
        """硬拆 thinking_content / content；无边界 token 时整段视为 content（Think-OFF 安全）。"""
        content_start = 0
        if self.think_end_id is not None:
            try:
                rev = gen_ids[::-1].index(self.think_end_id)
                content_start = len(gen_ids) - rev  # 结束标记之后第一个 token
            except ValueError:
                content_start = 0
        thinking = self.tokenizer.decode(
            gen_ids[:content_start], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            gen_ids[content_start:], skip_special_tokens=True
        ).strip("\n")
        return thinking, content, content_start

    @torch.no_grad()
    def generate_text(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool = False,
        sampling: dict | None = None,
    ) -> dict:
        """生成并硬拆 thinking/content（筛选阶段用）。返回 {full_text, thinking_content, content}。

        sampling 为 None 时用官方采样参数；传入 config.GREEDY_DECODING 可切换贪心（确定性）。
        """
        gen_kwargs = sampling if sampling is not None else config.sampling_kwargs(enable_thinking)
        prompt = self.build_prompt(messages, enable_thinking=enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens(enable_thinking),
            **gen_kwargs,
        )
        gen_ids = outputs[0][inputs.input_ids.shape[1]:].tolist()
        thinking, content, _ = self._split_thinking(gen_ids)
        full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return {
            "full_text": full_text,
            "thinking_content": thinking,
            "content": content,
        }

    @torch.no_grad()
    def generate_with_belief(
        self,
        messages: list[dict],
        *,
        choices: list[str] | None = None,
        enable_thinking: bool = False,
        sampling: dict | None = None,
        verbose: bool = False,
    ) -> dict:
        """生成 + thinking/content 硬拆 + 决策 token 信念探测（原始 logits）。

        sampling 为 None 时用官方采样参数；传入 config.GREEDY_DECODING 可切换贪心（确定性），
        用于受控 A/B 评测以消除采样噪声。
        """
        if choices is None:
            choices = config.CHOICES_MAX

        gen_kwargs = sampling if sampling is not None else config.sampling_kwargs(enable_thinking)
        prompt = self.build_prompt(messages, enable_thinking=enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens(enable_thinking),
            return_dict_in_generate=True,
            output_logits=True,  # 原始 logits，与采样温度解耦
            **gen_kwargs,
        )

        gen_ids = outputs.sequences[0][inputs.input_ids.shape[1]:].tolist()
        thinking, content, content_start = self._split_thinking(gen_ids)
        full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        empty_belief = {c: 1.0 / len(choices) for c in choices}
        if len(gen_ids) == 0:
            return {
                "messages": messages,
                "text": "",
                "content": "",
                "thinking_content": "",
                "full_text": "",
                "belief": empty_belief,
                "raw_logprobs": {c: math.log(empty_belief[c]) for c in choices},
                "target_token": "",
                "target_index": -1,
                "decision_method": "empty",
                "decision_valid": 0,
                "pred": "UNKNOWN",
                "parsed_answer": "UNKNOWN",
                "pred_matches_parse": 1,
            }

        token_strs = [self.tokenizer.decode([tid]) for tid in gen_ids]
        target_index, decision_method = locate_decision_token_index(
            token_strs, choices, search_start=content_start
        )

        # outputs.logits[i] 对应生成 gen_ids[i] 的原始 logits。
        # Gemma + 强干预时偶发：logits 元组短于 token 数 / target_index 越界。
        logits_steps = outputs.logits
        n_logits = len(logits_steps) if logits_steps is not None else 0
        if n_logits == 0:
            return {
                "messages": messages,
                "text": content,
                "content": content,
                "thinking_content": thinking,
                "full_text": full_text,
                "belief": empty_belief,
                "raw_logprobs": {c: math.log(empty_belief[c]) for c in choices},
                "target_token": "",
                "target_index": int(target_index),
                "decision_method": "no_logits",
                "decision_valid": 0,
                "pred": parse_answer(content, choices),
                "parsed_answer": parse_answer(content, choices),
                "pred_matches_parse": 1,
            }
        if target_index < 0 or target_index >= n_logits:
            # 回退到最后一个生成步，避免干预导致的 IndexError 中断整轮扫
            target_index = n_logits - 1
            decision_method = f"{decision_method}|logits_clamp"

        step_logits = logits_steps[target_index][0]
        probs = torch.softmax(step_logits.float(), dim=-1)

        raw_logprobs: dict[str, float] = {}
        for c in choices:
            p = sum(probs[tid].item() for tid in self._choice_token_ids.get(c, set()))
            raw_logprobs[c] = math.log(p + 1e-10)

        belief = softmax_from_logprobs(raw_logprobs)

        decision_token = token_strs[target_index]
        token_upper = decision_token.strip().upper()
        decision_valid = int(token_upper in choices)

        parsed = parse_answer(content, choices)
        pred = token_upper if decision_valid else parsed
        pred_matches_parse = int(pred == parsed)

        if verbose:
            print(
                f"DEBUG: 决策 token='{decision_token}' index={target_index} "
                f"method={decision_method} valid={bool(decision_valid)}"
            )
            print("thinking_content:", thinking[:120], "..." if len(thinking) > 120 else "")
            print("content:", content)
            print("belief:", belief)
            print(f"pred={pred} parse_answer={parsed} match={bool(pred_matches_parse)}")

        return {
            "messages": messages,
            "text": content,  # 向后兼容：text 即正式回答（不含 thinking）
            "content": content,
            "thinking_content": thinking,
            "full_text": full_text,
            "belief": belief,
            "raw_logprobs": raw_logprobs,
            "target_token": decision_token,
            "target_index": target_index,
            "decision_method": decision_method,
            "decision_valid": decision_valid,
            "pred": pred,
            "parsed_answer": parsed,
            "pred_matches_parse": pred_matches_parse,
        }
