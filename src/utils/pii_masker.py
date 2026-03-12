import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from src.utils.logger import logger
from src.core.exceptions import SecurityException

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False

class PIIMasker:
    """
    A utility class to detect and mask Personally Identifiable Information (PII)
    in text using Microsoft Presidio.
    
    Attributes:
        enabled (bool): Whether the Presidio engines are successfully loaded.
    """
    def __init__(self) -> None:
        """Initializes the PIIMasker with Presidio engines and a ThreadPoolExecutor for async tasks."""
        if HAS_PRESIDIO:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                self.enabled = True
                self._executor = ThreadPoolExecutor(max_workers=4)
                logger.info("[SEC_LAYER] Presidio PII Masker initialized.")
            except Exception as e:
                logger.warning(f"[SEC_LAYER] Presidio initialization failed: {e}")
                self.enabled = False
        else:
            logger.warning("[SEC_LAYER] Presidio not installed. PII Masking disabled.")
            self.enabled = False

    def mask_text(self, text: str) -> str:
        """
        Synchronously masks PII in the given text.
        
        Args:
            text (str): The raw input text.
            
        Returns:
            str: The anonymized text.
            
        Raises:
            SecurityException: If an error occurs during the masking process.
        """
        if not self.enabled or not text:
            return text
        try:
            results = self.analyzer.analyze(
                text=text, 
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"], 
                language='en'
            )
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized_result.text
        except Exception as e:
            logger.error(f"[SEC_LAYER] Error masking text: {e}", exc_info=True)
            raise SecurityException(f"PII Masking failed: {e}")

    async def mask_text_async(self, text: str) -> str:
        """
        Asynchronously masks PII using a ThreadPoolExecutor to prevent GIL blocking.
        
        Args:
            text (str): The raw input text.
            
        Returns:
            str: The anonymized text.
        """
        if not self.enabled or not text:
            return text
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.mask_text, text)

    async def mask_documents_async(self, texts: List[str]) -> List[str]:
        """
        Concurrently masks PII in a list of documents.
        
        Args:
            texts (List[str]): A list of raw strings.
            
        Returns:
            List[str]: A list of anonymized strings.
        """
        if not self.enabled:
            return texts
        tasks = [self.mask_text_async(text) for text in texts]
        return await asyncio.gather(*tasks)
