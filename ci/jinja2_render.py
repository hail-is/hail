import sys
import jinja2
import json


def jinja2_render(config, input, output, file_overrides):
    file_override(config, file_overrides)

    with open(input, 'r') as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    with open(output, 'w') as f:
        f.write(template.render(**config))


def file_override(config, file_overrides):
    assert set(config).issuperset(set(file_overrides))
    for k, v in file_overrides.items():
        if isinstance(v, dict):
            assert isinstance(config[k], dict), k
            file_override(config[k], v)
        else:
            assert isinstance(v, str)
            assert isinstance(config[k], str)
            config[k] = open(v).read()


def usage():
    print(f'usage: {sys.argv[0]} <json-literal-config> <input-file> <output-file> [<file-overrides>]', file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) not in (4, 5):
        usage()
    jinja2_render(json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else dict())


if __name__ == "__main__":
    main()
