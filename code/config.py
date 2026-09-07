# config.py
"""
集中式实验配置（Transformers 统一后端）。

code/ 下所有流水线脚本共享此处配置，命令行参数可覆盖对应默认值。
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

# 必须在首次 CUDA / cuBLAS 调用之前设置，否则确定性工作区不生效。
# 仅 import config 就会写入（各脚本均 import config）。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ===================== 模型与后端配置 =====================
# 本地模型权重目录（HuggingFace 格式）。请用环境变量或 CLI --model-path 覆盖，勿提交本机绝对路径。
#   export ANCHOR_MODEL_PATH=/path/to/Qwen3-8B
#   export ANCHOR_GEMMA_PATH=/path/to/gemma-4-12B-it
MODEL_PATH = os.environ.get("ANCHOR_MODEL_PATH", "").strip()
# 跨族复制（Gemma-4-12B-IT）常用路径
GEMMA4_12B_IT_PATH = os.environ.get("ANCHOR_GEMMA_PATH", "").strip()
# 输出目录中用于标识模型的短名（可与权重目录名不同）
MODEL_NAME = os.environ.get("ANCHOR_MODEL_NAME", "Qwen3-8B")

# 加载精度与放置策略（bfloat16 + 单卡 + sdpa）
DTYPE = torch.bfloat16
# 目标 GPU：单卡加载后 .to(DEVICE)。
# 可写 "cuda:0" / "cuda:1" / 整数 0/1 / "cpu"。
# ★ 换卡优先改这里；各脚本也可用 --device 覆盖。
# 注意：若同时设置 CUDA_VISIBLE_DEVICES=3，则进程内只能看到一张卡，应写 DEVICE="cuda:0"
# （可见卡的第 0 号），不要再写 cuda:3。两种方式二选一，勿叠加混淆。
DEVICE = "cuda:0"
ATTN_IMPLEMENTATION = "sdpa"  # H100 环境可改为 "flash_attention_2" 提速


def normalize_device(device) -> str:
    """把 0/'0'/'cuda:0' 统一成 LocalModel 可用的设备字符串。"""
    if device is None:
        return str(DEVICE)
    if isinstance(device, int):
        return f"cuda:{device}"
    s = str(device).strip()
    if s.isdigit():
        return f"cuda:{int(s)}"
    return s

# ===================== 生成与探测配置 =====================
# 是否允许模型输出 <think> 思维链（Qwen3 硬开关，透传给 apply_chat_template 的 enable_thinking）
ENABLE_THINK_MODE = True

# 采样配置：对齐 Qwen3 官方 generation_config.json 与最佳实践。
# 官方明确不建议在 think 模式使用贪心解码（易退化/无尽重复）。
THINK_SAMPLING = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}
NO_THINK_SAMPLING = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
}

# 分模式最大新 token 数：Think / No-Think 统一预算。
# No-Think 不再用提示词禁止分析，正文也可能较长推理，故与 Think 对齐。
MAX_NEW_TOKENS_THINK = 4096
MAX_NEW_TOKENS_NO_THINK = 4096

# 多轮实验中触发上下文溢出保护的 prompt token 阈值（None 表示按模型上限自动推断）
CONTEXT_TOKEN_LIMIT: int | None = None

# Qwen3 的 </think> token id（用于在 token 序列上硬拆 thinking / content）
THINK_END_TOKEN_ID = 151668

# 选项字母全集：MMLU/SciQ 用 A-D，CommonsenseQA 用 A-E（按题目动态截取）
CHOICES_MAX = ["A", "B", "C", "D", "E"]


# 贪心解码：确定性、可复现。用于受控评测（如干预 A/B 对比），消除采样噪声。
GREEDY_DECODING = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "min_p": None,
}


def sampling_kwargs(enable_thinking: bool) -> dict:
    """按 think 模式返回官方推荐的采样参数。"""
    return dict(THINK_SAMPLING if enable_thinking else NO_THINK_SAMPLING)


def max_new_tokens_for(enable_thinking: bool) -> int:
    return MAX_NEW_TOKENS_THINK if enable_thinking else MAX_NEW_TOKENS_NO_THINK

# ===================== 路径配置 =====================
# 以项目根目录为基准（本文件位于 code/ 下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE = PROJECT_ROOT / "raw_data" / "mmlu" / "dev-00000-of-00001.parquet"
FILTERED_DIR = PROJECT_ROOT / "filtered_data" / "mmlu_dev"
RESULTS_DIR = PROJECT_ROOT / "results"

# ===================== 复现性 =====================
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """固定随机种子，并尽量开启确定性算法。

    重要限制（请读）:
      - do_sample=False 只关闭「采样 RNG」，不保证 GPU 上 logits 比特级可复现。
      - bf16 + SDPA/FlashAttention 的归约顺序在跨进程/不同 launch 下仍可能抖动。
      - 多轮对话会把 R0 的微小文本差放大到后续轮。
      - 因此：同一测集扫 α/层时，请用 step3 --reuse-baseline-from 冻结 baseline。
      - 本函数应在 LocalModel / 首次 CUDA 张量创建之前调用（step3 已按此顺序）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        # warn_only：部分算子无确定性实现时只警告，避免直接崩溃
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
