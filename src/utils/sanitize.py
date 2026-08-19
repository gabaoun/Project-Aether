import re

_URL_RE = re.compile(r'https?://[^\s)>\]"\']+')


def strip_unsourced_links(answer: str, context: str) -> str:
    """
    Removes any URL in `answer` that doesn't appear verbatim in `context`, so
    the model can only echo links it actually retrieved - never invent,
    modify, or guess one. Leaves the surrounding text and trailing
    punctuation intact.
    """
    def _replace(match: re.Match) -> str:
        raw = match.group(0)
        url = raw.rstrip('.,;:!?')
        if url in context:
            return raw
        return raw[len(url):]

    return _URL_RE.sub(_replace, answer)
