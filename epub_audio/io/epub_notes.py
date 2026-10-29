"""Omit linked note markers from narration, preserving prose and note bodies."""
import posixpath
import re
from urllib.parse import unquote, urlsplit


_NOTE_CLASSES = {'footnote', 'noteref', 'footnoteref', 'footnote-ref', 'footnote-reference',
                 'endnoteref', 'endnote-ref', 'footnote-backref', 'footnote-back', 'backref'}
_MARKER = re.compile(r'(?:\[\s*(?:\d+|[ivxlcdm]+|[*†‡]+)\s*\]|\(\s*\d+\s*\)|'
                     r'\d+|[ivxlcdm]+|[*†‡]+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)', re.IGNORECASE)
_NOTE_ANCHOR = re.compile(r'(?:footnotes?|endnotes?|notes?|noteref|fnref|fn)[_:.\-]?\d+'
                          r'(?:[_:.\-].*)?', re.IGNORECASE)


def remove_note_references(soup):
    """Use EPUB semantics or note-link evidence, never a global number filter."""
    for link in soup.find_all('a'):
        types = set(str(link.get('epub:type', '')).split())
        roles = set(str(link.get('role', '')).split())
        if types & {'noteref', 'backlink'} or roles & {'doc-noteref', 'doc-backlink'}:
            link.decompose()
            continue
        label = link.get_text('', strip=True)
        if not _MARKER.fullmatch(label):
            continue
        try:
            target = urlsplit(str(link.get('href', '')))
        except ValueError:
            continue
        # Ordinary numeric links to websites or chapters are still narrated.
        if target.scheme or target.netloc or not target.fragment:
            continue
        classes = {value.casefold() for value in link.get('class', [])}
        fragment = unquote(target.fragment)
        filename = posixpath.basename(unquote(target.path)).casefold()
        note_file = re.fullmatch(r'(?:footnotes?|endnotes?|notes?)\.[^.]+', filename)
        bracketed_superscript = (label.startswith('[') and label.endswith(']')
                                 and (link.find('sup') is not None or link.find_parent('sup') is not None))
        if classes & _NOTE_CLASSES or _NOTE_ANCHOR.fullmatch(fragment) or note_file or bracketed_superscript:
            link.decompose()
