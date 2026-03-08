from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # 暂时不支持温度为 0 的贪婪搜索(greedy sampling)，为了代码精简统一走采样逻辑
        assert self.temperature > 1e-10, "greedy sampling is not permiited"