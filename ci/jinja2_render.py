import json
import sys

import jinja2


def jinja2_render(config, input, output):
    with open(input, 'r', encoding='utf-8') as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(template.render(**config))


def usage():
    print(f'usage: {sys.argv[0]} <json-literal-config> <input-file> <output-file>', file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) != 4:
        usage()
    jinja2_render(json.loads(sys.argv[1]), sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
