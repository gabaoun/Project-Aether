import re
import asyncio
from typing import List

class PIIMasker:
    """
    Masks PII (emails, phone numbers) from documents.
    """
    def __init__(self):
        self.email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_regex = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')

    def mask_text(self, text: str) -> str:
        text = self.email_regex.sub("[EMAIL]", text)
        text = self.phone_regex.sub("[PHONE]", text)
        return text

    async def mask_documents_async(self, texts: List[str]) -> List[str]:
        # Simulating async processing for heavy masking tasks
        loop = asyncio.get_event_loop()
        masked_texts = await asyncio.gather(*[loop.run_in_executor(None, self.mask_text, text) for text in texts])
        return masked_texts
