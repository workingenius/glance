from ocr_spl.model.model import SimpleCharOcr


def recognize_char(img) -> str:
    """recognize a single char from an small image"""
    socr = SimpleCharOcr()
    socr.load('data/chars.lib.d2hash.pkl')
    # currently return '.' if not recognized
    return socr.search(img, dist=100) or '.'
