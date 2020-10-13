from os import listdir
from os.path import isfile, join
from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader(['templates', 'pages']),
    autoescape=select_autoescape(['html', 'xml']))

pages = [f for f in listdir('pages') if isfile(join('pages', f))]
for fname in pages:
    print(f'rendering {fname}')
    page = env.get_template(fname)
    with open(f'www/{fname}', 'w') as f:
        f.write(page.render())
