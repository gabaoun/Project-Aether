import tiktoken
from src.utils.logger import logger

class TokenCounter:
    """
    Utility class to count tokens and estimate costs.
    """
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Approximate costs per 1M tokens
        self.cost_per_1m_input = 0.15  
        self.cost_per_1m_output = 0.60 

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def log_cost(self, operation: str, input_text: str, output_text: str = "") -> None:
        in_tokens = self.count_tokens(input_text)
        out_tokens = self.count_tokens(output_text)
        cost = (in_tokens / 1_000_000) * self.cost_per_1m_input + (out_tokens / 1_000_000) * self.cost_per_1m_output
        logger.info(f"[TOKEN_OPT] [{operation}] Cost: ${cost:.6f} | Tokens: {in_tokens} in / {out_tokens} out")
