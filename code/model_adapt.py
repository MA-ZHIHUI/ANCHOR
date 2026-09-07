# model_adapt.py
"""跨模型适配：解码器层路径、层数/隐维、think 边界 token。

当前覆盖:
  - Qwen3 CausalLM: model.model.layers；</think>
  - Gemma4 Unified: model.model.language_model.layers；Think-OFF 主用（无 Qwen 式 </think>）
"""

from __future__ import annotations

from typing import Any

# 候选 think 结束标记（按优先级尝试）
_THINK_END_CANDIDATES = (
    "</" + "think>",  # Qwen3
    "<channel|>",  # Gemma4 thought channel closer (template-dependent)
    "<|channel|>",
)


def get_text_config(model: Any):
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError("model has no config")
    text_cfg = getattr(cfg, "text_config", None)
    return text_cfg if text_cfg is not None else cfg


def get_decoder_layers(model: Any):
    """返回可挂 forward hook 的 Transformer block 列表（nn.ModuleList）。"""
    root = model.model if hasattr(model, "model") else model
    # Gemma4 Unified / 部分多模态壳
    lang = getattr(root, "language_model", None)
    if lang is not None and hasattr(lang, "layers"):
        return lang.layers
    # 标准 CausalLM（Qwen3 等）
    if hasattr(root, "layers"):
        return root.layers
    raise AttributeError(
        "无法定位 decoder layers（期望 model.model.layers 或 "
        "model.model.language_model.layers）"
    )


def get_num_hidden_layers(model: Any) -> int:
    return len(get_decoder_layers(model))


def get_hidden_size(model: Any) -> int:
    tc = get_text_config(model)
    if hasattr(tc, "hidden_size") and tc.hidden_size is not None:
        return int(tc.hidden_size)
    # 回退：读第 0 层输出维（需一次假输入时再考虑；此处尽量用 config）
    raise AttributeError("text_config.hidden_size 不可用")


def resolve_think_end_id(tokenizer) -> int | None:
    """解析用于硬拆 thinking/content 的结束 token id；找不到则返回 None。"""
    for tag in _THINK_END_CANDIDATES:
        try:
            tid = tokenizer.convert_tokens_to_ids(tag)
        except Exception:
            tid = None
        if tid is not None and isinstance(tid, int) and tid >= 0:
            # 过滤 unk
            unk = getattr(tokenizer, "unk_token_id", None)
            if unk is not None and tid == unk:
                continue
            return int(tid)
        # 部分特殊符号不在 convert_tokens_to_ids，用 encode
        try:
            enc = tokenizer.encode(tag, add_special_tokens=False)
            if len(enc) == 1:
                return int(enc[0])
        except Exception:
            pass
    return None


def model_family_tag(model_path: str) -> str:
    s = model_path.lower()
    if "gemma" in s:
        return "gemma"
    if "qwen" in s:
        return "qwen"
    return "unknown"
