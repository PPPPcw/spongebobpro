import json
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集：加载 preprocess_data.py 生成的 .bin + .meta
    .bin: uint16 tokens，shape=(num_chunks, seq_len)
    """

    def __init__(self, data_path: str, seq_len: int = 512):
        # 兼容 pretrain.py 里默认值带花括号的写法："{/abs/path.bin}"
        data_path = (data_path or "").strip()
        if data_path.startswith("{") and data_path.endswith("}"):
            data_path = data_path[1:-1].strip()

        # 兼容用户传入不带后缀
        if not (data_path.endswith(".bin") or data_path.endswith(".meta")):
            data_path = data_path + ".bin"

        # 若误传 .meta，则映射回 .bin
        if data_path.endswith(".meta"):
            data_path = os.path.splitext(data_path)[0] + ".bin"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}")

        meta_path = os.path.splitext(data_path)[0] + ".meta"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"找不到元信息文件: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # 允许传入 seq_len 做一致性检查（不一致就直接报错，避免 silent bug）
        if int(self.meta.get("seq_len", seq_len)) != int(seq_len):
            raise ValueError(f"seq_len 不匹配：meta={self.meta.get('seq_len')} vs args={seq_len}")

        self.data_path = data_path
        self.meta_path = meta_path
        self.seq_len = int(seq_len)

        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode="r",
            shape=tuple(self.meta["shape"]),
        )

    def __len__(self) -> int:
        return int(self.meta["num_chunks"])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(self.data[idx].astype(np.int64))
        return chunk.clone(), chunk.clone()