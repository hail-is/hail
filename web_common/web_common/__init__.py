from .web_common import (
    base_context,
    render_template,
    sass_compile,
    set_message,
    setup_aiohttp_jinja2,
    setup_common_static_routes,
)

__all__ = [
    'sass_compile',
    'setup_aiohttp_jinja2',
    'setup_common_static_routes',
    'set_message',
    'base_context',
    'render_template',
]
