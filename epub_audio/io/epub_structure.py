"""Read EPUB navigation and headings without dropping unlisted narrative text."""
from dataclasses import dataclass
import posixpath
import re
import unicodedata
from urllib.parse import unquote, urlsplit
from epub_audio.core.models import BookMetadata, Chapter
from epub_audio.core.text import has_spoken_text, is_scene_break
from epub_audio.config.defaults import SCENE_BREAK
from epub_audio.io.epub_notes import remove_note_references


def clean(value):
    return re.sub(r'\s+', ' ', value or '').strip()


def _identity(value):
    return ''.join(character for character in unicodedata.normalize('NFKD', value).casefold()
                   if character.isalnum())


def read_metadata(book):
    def values(name):
        return book.get_metadata('DC', name)
    def first(name):
        return next((clean(value) for value, _ in values(name) if clean(value)), '')
    authors, sorts = [], []
    for value, attributes in values('creator'):
        role = next((v for k, v in attributes.items() if k.split('}')[-1] == 'role'), 'aut')
        if role != 'aut' or not clean(value):
            continue
        authors.append(clean(value))
        sorts.append(next((clean(v) for k, v in attributes.items()
                           if k.split('}')[-1] == 'file-as' and clean(v)), ''))
    dates = sorted(values('date'), key=lambda item: not any(
        k.split('}')[-1] == 'event' and v == 'publication' for k, v in item[1].items()))
    year = next((m.group(1) for value, _ in dates
                 if (m := re.match(r'^([12]\d{3})(?:\D|$)', clean(value)))), '')
    properties = {}
    for _, attributes in book.get_metadata('OPF', 'meta'):
        key = attributes.get('name') or attributes.get('property')
        if key:
            properties.setdefault(key, attributes.get('content', ''))
    series = clean(properties.get('calibre:series'))
    series_index = clean(properties.get('calibre:series_index'))
    # EPUB 3 collection metadata, including refinements on a collection ID.
    if not series:
        for value, attributes in book.get_metadata('OPF', 'meta'):
            if attributes.get('property') == 'belongs-to-collection':
                reference = '#' + attributes.get('id', '')
                refinements = {a.get('property'): clean(v) for v, a in book.get_metadata('OPF', 'meta')
                               if a.get('refines') == reference}
                if refinements.get('collection-type', 'series') == 'series':
                    series = clean(value)
                    series_index = refinements.get('group-position', '')
                    break
    return BookMetadata(first('title'), authors, sorts, year, series, series_index)


@dataclass
class Block:
    text: str
    level: int = 0


@dataclass
class Document:
    name: str
    title: str
    blocks: list
    anchors: dict
    technical: bool


def read_document(item, metadata, split_anchors=()):
    from bs4 import BeautifulSoup, Comment, NavigableString
    soup = BeautifulSoup(item.get_content(), 'html.parser')
    for node in soup(['script', 'style']):
        node.decompose()
    remove_note_references(soup)
    body = soup.body or soup
    heading = body.find(['h1', 'h2'])
    title = heading.get_text(' ', strip=True) if heading else item.get_name()
    blocks, buffer, anchors = [], [], {}
    level = 0

    def flush():
        text = clean(''.join(buffer))
        buffer.clear()
        if has_spoken_text(text) or is_scene_break(text):
            blocks.append(Block(text, level))

    def visit(node):
        nonlocal level
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            buffer.append(str(node))
            return
        if node.name == 'nav' and 'toc' in str(node.get('epub:type', '')).split():
            return
        block = node.name in ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li',
                               'blockquote', 'div', 'br', 'tr', 'td')
        if block or node.get('id') in split_anchors or (node.name == 'a' and node.get('name') in split_anchors):
            flush()
        for anchor in (node.get('id'), node.get('name') if node.name == 'a' else None):
            if anchor:
                anchors.setdefault(unquote(anchor), len(blocks))
        if node.name == 'hr':
            flush()
            blocks.append(Block(SCENE_BREAK))
            return
        previous = level
        if re.fullmatch(r'h[1-6]', node.name or ''):
            level = int(node.name[1])
        for child in node.children:
            visit(child)
        if block:
            flush()
        level = previous

    visit(body)
    flush()
    texts = [block.text for block in blocks]
    text = ' '.join(texts)
    types = set()
    # A technical section nested inside a story must not hide the entire file.
    roots = [soup.html, body]
    children = [node for node in body.children if getattr(node, 'name', None)]
    if len(children) == 1:
        roots.append(children[0])
    for node in roots:
        if node:
            types.update(str(node.get('epub:type', '')).split())
    technical = bool(types & {'cover', 'titlepage', 'copyright-page', 'toc', 'colophon'})
    filename = posixpath.basename(item.get_name()).casefold()
    technical |= bool(re.match(r'^(cover|titlepage|copyright|colophon|toc|nav)([._-]|$)', filename))
    copyright_notice = ('©' in text or bool(re.search(r'(?i)\bcopyright\s+(?:©\s*)?\d{4}', text)))
    if len(text) < 2000 and not any(block.level for block in blocks) and copyright_notice:
        technical |= any(term in text.casefold() for term in
                         ('titre original', 'droits réservés', 'all rights reserved', 'copyright'))
    if metadata.title and metadata.authors and len(text) < 800 and len(blocks) <= 16:
        author_names = []
        for author in metadata.authors:
            family, separator, given = author.partition(',')
            author_names.append(given + ' ' + family if separator else author)
        technical |= (_identity(metadata.title) in _identity(text)
                      and any(_identity(author) in _identity(text) for author in author_names)
                      and all(len(block.text) < 150 for block in blocks)
                      and not any(block.level and _identity(block.text) != _identity(metadata.title)
                                  for block in blocks))
    return Document(item.get_name(), title, blocks, anchors, technical)


def _links(nodes):
    for node in nodes:
        if isinstance(node, (tuple, list)) and len(node) == 2 and isinstance(node[1], (list, tuple)):
            yield node[0]
            yield from _links(node[1])
        elif hasattr(node, 'href'):
            yield node


def navigation_fragments(book):
    return {unquote(urlsplit(link.href).fragment) for link in _links(book.toc)
            if getattr(link, 'href', '') and urlsplit(link.href).fragment}


def plan_chapters(book, documents, navigation_error=''):
    """Return modern chapters, or the exact legacy chapter list and a reason."""
    legacy = [Chapter(doc.title, [block.text for block in doc.blocks],
                      tuple(i for i, block in enumerate(doc.blocks) if block.level)) for doc in documents
              if any(has_spoken_text(block.text) for block in doc.blocks)]
    content = [doc for doc in documents if not doc.technical and doc.blocks]
    excluded = len(documents) - len(content)
    blocks, starts = [], {}
    for doc in content:
        starts[doc.name] = len(blocks)
        blocks.extend(doc.blocks)
    all_docs = {doc.name: doc for doc in documents}

    def fallback(reason):
        return legacy, 'legacy', reason, 0

    if navigation_error:
        return fallback(navigation_error)
    if not blocks:
        return fallback('Aucun récit identifiable après les pages de présentation.')
    links = list(_links(book.toc))
    boundaries = []
    if links:
        bases = [''] + [posixpath.dirname(item.get_name()) for item in book.get_items()
                        if item.media_type == 'application/x-dtbncx+xml'
                        or 'nav' in getattr(item, 'properties', [])]
        for link in links:
            href = getattr(link, 'href', '')
            if not href:
                # A section without a target only groups its children.
                continue
            target = urlsplit(href)
            if target.scheme or target.netloc:
                return fallback('La table des matières contient un lien extérieur au livre.')
            name = next((posixpath.normpath(posixpath.join(base, unquote(target.path)))
                         for base in bases
                         if posixpath.normpath(posixpath.join(base, unquote(target.path))) in all_docs), None)
            if name is None:
                return fallback('Un lien de la table des matières ne correspond à aucun document.')
            doc = all_docs[name]
            if doc.technical or not doc.blocks:
                continue
            fragment = unquote(target.fragment)
            if fragment and fragment not in doc.anchors:
                return fallback('Une ancre de la table des matières est introuvable.')
            offset = doc.anchors[fragment] if fragment else 0
            if offset >= len(doc.blocks):
                return fallback('Un chapitre pointe après la fin du texte de son document.')
            position = starts[name] + offset
            label = clean(getattr(link, 'title', '')) or doc.title
            if boundaries and position < boundaries[-1][0]:
                return fallback('La table des matières contredit l’ordre de lecture du livre.')
            if boundaries and position == boundaries[-1][0]:
                # Parent and first child may point to the same anchor.
                continue
            boundaries.append((position, label))
        mode = 'table des matières'
    else:
        boundaries = [(index, block.text) for index, block in enumerate(blocks) if block.level in (1, 2)]
        if not boundaries:
            return fallback('Table des matières absente et titres de chapitres non identifiables.')
        mode = 'titres'
    if not boundaries:
        return fallback('La table des matières ne définit aucun chapitre contenant du récit.')
    if any(has_spoken_text(block.text) for block in blocks[:boundaries[0][0]]):
        return fallback('Du texte non classé précède le premier chapitre : conservation nécessaire.')
    boundaries[0] = (0, boundaries[0][1])
    chapters, pending, pending_labels = [], [], []
    for index, (start, title) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(blocks)
        section = blocks[start:end]
        only_label = clean(' '.join(block.text for block in section)).casefold() == title.casefold()
        if only_label or not any(block.level == 0 and has_spoken_text(block.text) for block in section):
            pending.extend(section)
            pending_labels.append(title)
            continue
        paragraphs = [block.text for block in pending + section]
        chapter_title = ' — '.join(pending_labels + [title])
        chapters.append(Chapter(chapter_title, paragraphs,
                                tuple(i for i, block in enumerate(pending + section) if block.level)))
        pending, pending_labels = [], []
    if pending or not chapters:
        return fallback('La structure contient une section finale sans récit identifiable.')
    # A hard invariant: modern planning must preserve every non-technical text block,
    # including documents absent from the TOC (continuations, notes, prefaces).
    if [text for chapter in chapters for text in chapter.paragraphs] != [block.text for block in blocks]:
        return fallback('Le contrôle de conservation du texte a échoué.')
    return chapters, mode, '', excluded
