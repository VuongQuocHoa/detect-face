import sys
from html.parser import HTMLParser

VOID_TAGS = {
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'link', 'meta', 'param', 'source', 'track', 'wbr'
}

class TagValidator(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in VOID_TAGS:
            self.stack.append((tag.lower(), self.getpos()))

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"Unexpected closing tag </{tag_lower}> at line {self.getpos()[0]} (no open tag)")
            return
        last_tag, pos = self.stack.pop()
        if last_tag != tag_lower:
            self.errors.append(f"Mismatched tag: expected </{last_tag}> (opened at line {pos[0]}), but got </{tag_lower}> at line {self.getpos()[0]}")

def check_file(filepath):
    print(f"\n==========================================")
    print(f" Checking: {filepath}")
    print(f"==========================================")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    v = TagValidator()
    v.feed(content)

    if v.errors:
        print(f"Found {len(v.errors)} mismatch/unexpected errors:")
        for err in v.errors:
            print(f"  [X] {err}")
    else:
        print("  [V] No mismatched or unexpected tags.")

    if v.stack:
        print(f"Found {len(v.stack)} unclosed tags:")
        for tag, pos in v.stack:
            print(f"  [X] <{tag}> opened at line {pos[0]} was NOT closed!")
    else:
        print("  [V] All opening tags are closed.")

    if not v.errors and not v.stack:
        print("\n===> 100% PERFECT! Every tag opens and closes properly!")

if __name__ == '__main__':
    check_file('d:/PROJECTS/detect-face/index.html')
    check_file('d:/PROJECTS/detect-face/app/templates/index.html')
