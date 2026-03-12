import tiktoken
from src.utils.logger import logger

class TokenCounter:
    """
    A utility class to count tokens and calculate approximate costs for OpenAI models.
    
    Attributes:
        model_name (str): The name of the OpenAI model to use for encoding.
        cost_per_1m_input (float): Cost per 1 million input tokens.
        cost_per_1m_output (float): Cost per 1 million output tokens.
    """
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        """
        Initializes the TokenCounter with a specific model.
        
        Args:
            model_name (str): The model name (default: "gpt-4o-mini").
        """
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        # Approximate costs per 1M tokens
        self.cost_per_1m_input = 0.15  
        self.cost_per_1m_output = 0.60 

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text.
        
        Args:
            text (str): The input string to tokenize.
            
        Returns:
            int: The number of tokens.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def log_cost(self, operation: str, input_text: str, output_text: str = "") -> None:
        """
        Calculates and logs the estimated cost of an LLM operation.
        
        Args:
            operation (str): A descriptive name for the operation (e.g., "MetadataEnrichment").
            input_text (str): The prompt text sent to the model.
            output_text (str): The response text received from the model.
        """
        in_tokens = self.count_tokens(input_text)
        out_tokens = self.count_tokens(output_text)
        cost = (in_tokens / 1_000_000) * self.cost_per_1m_input + (out_tokens / 1_000_000) * self.cost_per_1m_output
        logger.info(f"[TOKEN_OPT] [{operation}] Cost: ${cost:.6f} | Tokens: {in_tokens} in / {out_tokens} out")
