import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str # 模型在本地的路径
    max_num_batched_tokens: int = 16384 # 单次推理最大处理的token总数
    max_num_seqs: int = 512 # 最大并发请求数
    max_model_len: int = 4096 # 单个请求的最大上下文长度
    gpu_memory_utilization: float = 0.9 # KV Cache 显存利用率 (默认占用剩余显存的 90%)
    tensor_parallel_size: int = 1 # 张量并行度 (使用的 GPU 数量)
    enforce_eager: bool = False # 是否强制禁用 CUDA Graph 优化
    hf_config: AutoConfig | None = None # 存放 HuggingFace 的配置
    eos: int = -1 # 结束符 ID
    kvcache_block_size: int = 256 # PagedAttention 的 Block 大小 (每个 block 存 256 个 token)
    num_kvcache_blocks: int = -1 # 物理块总数 (后续会根据 GPU 显存动态计算)
    
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 利用 transformers 库自动加载模型的 config.json
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 确保设置的最大长度不超过模型本身支持的极限
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len