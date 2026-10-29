"""Prepare spoken chapter headings using EPUB heading metadata."""
import re
import unicodedata

_ROMAN = re.compile(r'M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})')
_HEADING = re.compile(r'(?:(?P<label>(?i:chapitre))\s+)?(?P<number>[IVXLCDMivxlcdmⅠ-ⅿ]+)(?P<suffix>\s*(?:[.:\-–—)].*)?)', re.DOTALL)
_SMALL = ('zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit',
          'neuf', 'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize')
_TENS = {20:'vingt', 30:'trente', 40:'quarante', 50:'cinquante', 60:'soixante'}
_HEADING_END = re.compile(r'''[.!?…:;][\s»”"’')\]}]*$''')


def roman_number(text):
    numeral = unicodedata.normalize('NFKC', text).upper()
    if not numeral or _ROMAN.fullmatch(numeral) is None:
        return None
    values = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    total, previous = 0, 0
    for letter in reversed(numeral):
        value = values[letter]
        total += -value if value < previous else value
        previous = value
    return total


def french_number(number):
    """Cardinal chapter numbers from 1 to 3999, without an extra dependency."""
    if type(number) is not int or not 1 <= number <= 3999:
        raise ValueError('Numéro de chapitre hors plage.')
    if number <= 16:
        return _SMALL[number]
    if number < 20:
        return 'dix-' + _SMALL[number-10]
    if number < 70:
        ten, unit = divmod(number, 10)
        base = _TENS[ten*10]
        return base if not unit else base + (' et un' if unit == 1 else '-' + _SMALL[unit])
    if number < 80:
        return 'soixante et onze' if number == 71 else 'soixante-' + french_number(number-60)
    if number < 100:
        return 'quatre-vingts' if number == 80 else 'quatre-vingt-' + french_number(number-80)
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        base = 'cent' if hundreds == 1 else _SMALL[hundreds] + ' cent'
        if rest:
            return base + ' ' + french_number(rest)
        return base + ('s' if hundreds > 1 else '')
    thousands, rest = divmod(number, 1000)
    base = 'mille' if thousands == 1 else _SMALL[thousands] + ' mille'
    return base + (' ' + french_number(rest) if rest else '')


def spoken_roman_heading(text):
    match = _HEADING.fullmatch(text.strip())
    if match is None:
        return text
    numeral = match.group('number')
    # Mixed-case words such as "Mix" are not bare Roman chapter labels.
    if match.group('label') is None and not numeral.isupper():
        return text
    number = roman_number(numeral)
    if number is None:
        return text
    return 'Chapitre ' + french_number(number) + match.group('suffix')


def normalize_chapter_numbers(paragraphs, heading_indices=()):
    """Change at most one number in the consecutive opening headings, never prose."""
    output = list(paragraphs)
    headings = set(heading_indices)
    for index, text in enumerate(output):
        if not text.strip():
            continue
        if index not in headings:
            break
        converted = spoken_roman_heading(text)
        if converted != text:
            output[index] = converted
            break
    return output


def punctuate_chapter_headings(paragraphs, heading_indices=()):
    """Terminate EPUB headings before synthesis; preserve prose and punctuation."""
    headings = set(heading_indices)
    return [text.rstrip() + '.'
            if index in headings and any(character.isalnum() for character in text)
            and _HEADING_END.search(text.rstrip()) is None else text
            for index, text in enumerate(paragraphs)]


def prepare_chapter_headings(paragraphs, heading_indices=()):
    """Speak chapter numbers, soften all-caps headings and terminate their text."""
    headings = set(heading_indices)
    output = normalize_chapter_numbers(paragraphs, headings)
    for index, text in enumerate(output):
        if index not in headings or not text.isupper() or roman_number(text.strip()) is not None:
            continue
        # Keep mixed-case titles as written; capitalize the first letter even
        # when an opening quotation mark or a chapter number precedes it.
        output[index] = re.sub(r'[^\W\d_]', lambda match: match[0].upper(), text.lower(), count=1)
    return punctuate_chapter_headings(output, headings)
