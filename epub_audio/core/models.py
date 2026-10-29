"""Data exchanged between reading and processing."""
from dataclasses import dataclass, field


@dataclass
class Chapter:
    title: str
    paragraphs: list[str]
    heading_indices: tuple[int, ...] = ()


@dataclass
class BookMetadata:
    title: str = ''
    authors: list[str] = field(default_factory=list)
    author_sorts: list[str] = field(default_factory=list)
    year: str = ''
    series: str = ''
    series_index: str = ''

    def introduction(self):
        if not self.title:
            return ''
        text = self.title.rstrip('.!?…') + '.'
        if self.authors:
            names = []
            for author in self.authors:
                family, separator, given = author.partition(',')
                names.append((given.strip() + ' ' + family.strip()) if separator else author)
            text += ' Un livre de ' + ' et '.join(names).rstrip('.') + '.'
        return text


@dataclass
class BookPlan:
    chapters: list[Chapter]
    cover: object = None
    metadata: BookMetadata = field(default_factory=BookMetadata)
    mode: str = 'structure'
    fallback_reason: str = ''
    excluded_pages: int = 0
    chapter_analysis: dict = field(default_factory=dict)

    def __iter__(self):
        # Preserve the reader's existing chapters, cover unpacking API.
        yield self.chapters
        yield self.cover
