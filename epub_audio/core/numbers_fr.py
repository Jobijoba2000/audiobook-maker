"""French spoken numbers, applied to narration without rewriting the EPUB."""
import re
from num2words import num2words

_NUMBER = re.compile(
    r'(?<![\w.,])(?P<sign>[-−+]?)'
    r'(?P<integer>\d{1,3}(?:[ \u00a0\u202f\u2009]\d{3})+'
    r'|\d{1,3}(?:\.\d{3})+|\d+)'
    r'(?P<fraction>[,.]\d+)?'
    r'(?P<ordinal>ème|eme|er|re|e|ᵉ)?(?!\w)', re.IGNORECASE)


def spoken_numbers(text):
    """Read French thousands, years, signed numbers, decimals and ordinals.

    Decimal digits are read separately to retain zeros. Exact groups of three
    separated by dots are interpreted as thousands (legacy French typography).
    Embedded identifiers (B612), URLs and email addresses are kept intact.
    """
    protected = re.compile(r'https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
    def convert(match):
        digits = re.sub(r'[. \u00a0\u202f\u2009]', '', match['integer'])
        try:
            value = int(digits)
            if match['ordinal'] and not match['fraction']:
                result = num2words(value, lang='fr', to='ordinal')
                if value == 1 and match['ordinal'].lower() == 're':
                    result = 'première'
            else:
                if len(digits) > 1 and digits.startswith('0'):
                    result = ' '.join(num2words(int(digit), lang='fr') for digit in digits)
                else:
                    result = num2words(value, lang='fr')
                if match['fraction']:
                    result += ' virgule ' + ' '.join(
                        num2words(int(digit), lang='fr') for digit in match['fraction'][1:])
            if match['sign'] in ('-', '−'):
                result = 'moins ' + result
            elif match['sign'] == '+':
                result = 'plus ' + result
            return result
        except (ValueError, OverflowError):
            return match[0]
    pieces = []
    start = 0
    for match in protected.finditer(text):
        pieces.extend((_NUMBER.sub(convert, text[start:match.start()]), match[0]))
        start = match.end()
    pieces.append(_NUMBER.sub(convert, text[start:]))
    return ''.join(pieces)
