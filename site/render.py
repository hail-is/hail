from hailtop.hail_logging import configure_logging
import sass
import mistune
import importlib
import shutil
import web_common
import logging
import os
from os import listdir
from os.path import isfile, join
from jinja2 import (Environment, FileSystemLoader, select_autoescape, Template,
                    PackageLoader, ChoiceLoader)
from jinja2.utils import concat

configure_logging()
log = logging.getLogger('render')

site_env = Environment(
    loader=FileSystemLoader(['templates', 'pages']),
    autoescape=select_autoescape(['html', 'xml']))

log.info('>> rendering static pages <<')

pages = [f for f in listdir('pages') if isfile(join('pages', f))]
os.makedirs('site', exist_ok=True)
for fname in pages:
    log.info(f'rendering {fname}')
    page = site_env.get_template(fname)
    with open(f'site/{fname}', 'w') as f:
        f.write(page.render())

log.info('>> rendering dev docs <<')

docs_env = Environment(
    loader=ChoiceLoader([PackageLoader('web_common'),
                         FileSystemLoader(['docs-templates', 'docs-pages', 'dev-docs'])]),
    autoescape=False)

pages = [f for f in listdir('docs-pages') if isfile(join('docs-pages', f))]
os.makedirs('docs', exist_ok=True)
for fname in pages:
    log.info(f'rendering {fname}')
    page = docs_env.get_template(fname)
    with open(f'docs/{fname}', 'w') as f:
        f.write(page.render(userdata=None))

markdown = mistune.Markdown()

for dirpath, dirnames, filenames in os.walk('dev-docs'):
    target_dirpath = 'docs/' + dirpath
    os.makedirs(target_dirpath, exist_ok=True)

    path_links = []
    prefix = ''
    for superdir in dirpath.split('/'):
        prefix += '/' + superdir
        template = docs_env.get_template(prefix[len('dev-docs/'):] + '/index.md')
        folder_name = concat(template.blocks['title'](template.new_context())).strip()
        path_links.append((folder_name, prefix))
    log.info(path_links)

    file_links = []
    for fname in filenames:
        source = dirpath + '/' + fname
        extension = fname.split('.')[-1]
        name_without_extension = fname[:-(len(extension) + 1)]
        target = target_dirpath + '/' + name_without_extension + '.html'

        if fname == 'index.md':
            continue
        elif extension == 'md':
            log.info(f'{source} -> {target}')
            try:
                with open(target, 'w') as output:
                    template = docs_env.get_template(source[len('dev-docs/'):])
                    markdown_content =  concat(template.blocks['docs_content'](template.new_context()))
                    rendered_markdown = markdown(markdown_content)
                    def docs_content(context):
                        return rendered_markdown
                    template.blocks['docs_content'] = docs_content
                    page_name = concat(template.blocks['title'](template.new_context())).strip()
                    output.write(template.render(
                        userdata=None,
                        path=path_links))
            except FileNotFoundError as err:
                log.warning('skipping {source}', exc_info=True)
            file_links.append((page_name, name_without_extension + '.html'))

    source = dirpath + '/index.md'
    target = target_dirpath + '/' + 'index.html'
    log.info(f'{source} -> {target}')

    dir_links = []
    for dirname in dirnames:
        template = docs_env.get_template(dirpath[len('dev-docs/'):] + '/' + dirname + '/index.md')
        folder_name = concat(template.blocks['title'](template.new_context())).strip()
        dir_links.append((folder_name, dirname))
    with open(target, 'w') as output:
        template = docs_env.get_template(source[len('dev-docs/'):])
        page_name = concat(template.blocks['title'](template.new_context())).strip()
        output.write(template.render(
            path=path_links[:-1],
            files=file_links,
            folders=dir_links,
            userdata=None))


web_common.sass_compile('web_common')
module = importlib.import_module('web_common')
module_root = os.path.dirname(os.path.abspath(module.__file__))
shutil.rmtree('docs/common_static')
os.makedirs('docs/common_static', exist_ok=True)
shutil.copytree(module_root + '/static/css', 'docs/common_static/css')

sass.compile(
    dirname=('docs-styles', 'docs/common_static/css'), output_style='compressed',
    include_paths=[f'/Users/dking/projects/hail/web_common/styles'])  # FIXME
