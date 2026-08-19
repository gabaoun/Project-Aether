import asyncio
import re


class PIIMasker:
    """
    Masks PII (emails, phone numbers) from documents.
    """
    def __init__(self):
        self.email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        # Two alternatives: international "+<country> <area> <local>" numbers
        # (e.g. Brazilian mobiles: +55 84 99958-2391, which don't fit a 3-3-4
        # grouping), and the original North-American 3-3-4 style without a
        # leading '+'. Keeping them separate (rather than one loose "8+ digits
        # with separators" pattern) avoids masking date ranges like "2019-2023".
        self.phone_regex = re.compile(
            r'\+\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{0,4}'
            r'|\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        )
        # Brazilian CPF, canonical punctuated form only (XXX.XXX.XXX-XX).
        # Bare 11-digit runs aren't matched - too easy to collide with
        # unrelated numeric IDs and produce false positives.
        self.cpf_regex = re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b')

    def mask_text(self, text: str) -> str:
        text = self.email_regex.sub("[EMAIL]", text)
        text = self.phone_regex.sub("[PHONE]", text)
        text = self.cpf_regex.sub("[CPF]", text)
        return text

    async def mask_documents_async(self, texts: list[str]) -> list[str]:
        # Simulating async processing for heavy masking tasks
        loop = asyncio.get_event_loop()
        masked_texts = await asyncio.gather(*[loop.run_in_executor(None, self.mask_text, text) for text in texts])
        return masked_texts
