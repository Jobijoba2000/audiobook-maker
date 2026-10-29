"""EPUB reading order, HTML text extraction and cover files."""
from epub_audio.core.models import BookPlan
from epub_audio.io.epub_structure import read_metadata, read_document, plan_chapters, navigation_fragments


def read_book(path):
    try:
        import ebooklib
        from ebooklib import epub
        import bs4
        from lxml import etree
    except ImportError as exc:
        raise RuntimeError('Dépendances EPUB absentes. Lancez start_ebook.bat.') from exc
    class Reader(epub.EpubReader):
        """Confine damaged navigation to the automatic legacy-reading fallback."""
        navigation_error = ''

        def _parse_ncx(self, data):
            try:
                return super()._parse_ncx(data)
            except (etree.Error, IndexError, AttributeError, KeyError, ValueError):
                self.book.toc = ()
                self.navigation_error = 'La table des matières NCX est mal formée.'

        def _parse_nav(self, data, base_path, navtype='toc'):
            try:
                return super()._parse_nav(data, base_path, navtype)
            except (etree.Error, IndexError, AttributeError, KeyError, ValueError):
                if navtype == 'toc':
                    self.book.toc = ()
                    self.navigation_error = 'La table des matières EPUB est mal formée.'

        def _load_spine(self):
            try:
                return super()._load_spine()
            except (AttributeError, KeyError, epub.EpubException):
                if not self.book.spine:
                    raise
                self.book.toc = ()
                self.navigation_error = 'La référence à la table des matières est invalide.'

    # Prefer EPUB 3 navigation, or NCX when no nav exists. Close the archive
    # explicitly, including when a corrupt EPUB raises during its loading.
    with open(path, 'rb') as stream:
        reader = Reader(stream, options={'ignore_ncx': True})
        try:
            book = reader.load()
        finally:
            if reader.zf is not None:
                reader.zf.close()
    metadata = read_metadata(book)
    anchors = navigation_fragments(book)
    documents = []
    # The reading order is the EPUB spine, not ZIP storage order.
    items = [book.get_item_with_id(item_id) for item_id, linear in book.spine if linear != 'no']
    if not items:
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    for item in items:
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT or 'nav' in getattr(item, 'properties', []):
            continue
        documents.append(read_document(item, metadata, anchors))
    chapters, mode, reason, excluded = plan_chapters(book, documents, reader.navigation_error)
    if not chapters:
        raise RuntimeError('Aucun chapitre contenant du texte dans cet EPUB.')
    images = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))
    covers = list(book.get_items_of_type(ebooklib.ITEM_COVER))
    cover_ids = [metadata.get('content') for _, metadata in book.get_metadata('OPF', 'cover')]
    preferred = [book.get_item_with_id(key) for key in cover_ids if key]
    cover = next((item for item in preferred + covers + [i for i in images if 'cover' in i.get_name().lower()] + images
                  if item is not None and item.get_type() in (ebooklib.ITEM_COVER, ebooklib.ITEM_IMAGE)), None)
    return BookPlan(chapters, cover, metadata, mode, reason, excluded)


def save_cover(cover, directory):
    if cover is None:
        return
    data = cover.get_content()
    extension = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/gif': '.gif',
                 'image/webp': '.webp', 'image/svg+xml': '.svg'}.get(cover.media_type)
    if extension:
        (directory / ('cover' + extension)).write_bytes(data)
