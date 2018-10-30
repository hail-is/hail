from IPython.display import display
from ipywidgets import widgets, link, Layout

import hail as hl

__all__ = [
    'interact',
]


def interact(obj, cache=True):
    if cache:
        obj = obj.cache()
    tab = widgets.Tab()
    base_style = widgets.ButtonStyle()
    selected_style = widgets.ButtonStyle(button_color='#DDFFDD', font_weight='bold')

    if isinstance(obj, hl.Table):
        glob = widgets.Button(description='globals',
                              layout=widgets.Layout(width='120px', height='30px'))
        rows = widgets.Button(description='rows',
                              layout=widgets.Layout(width='120px', height='200px'))
        rows.style = selected_style

        base = hl.utils.LinkedList(str).push('ht')
        tab.children = [recursive_build(obj.globals.dtype, base, 'mh.globals'),
                        recursive_build(obj.row.dtype, base, 'ht.row')]
        tab.set_title(0, 'globals')
        tab.set_title(1, 'row')
        tab.selected_index = 1
        selection_handler = widgets.IntText(1)

        box = widgets.VBox([glob, rows])
        buttons = [glob, rows]
    else:
        assert isinstance(obj, hl.MatrixTable)
        glob = widgets.Button(description='globals',
                              layout=widgets.Layout(width='65px', height='30px'))
        cols = widgets.Button(description='cols',
                              layout=widgets.Layout(width='200px', height='30px'))
        rows = widgets.Button(description='rows',
                              layout=widgets.Layout(width='65px', height='200px'))
        entries = widgets.Button(description='entries',
                                 layout=widgets.Layout(width='200px', height='200px'))
        entries.style = selected_style

        base = hl.utils.LinkedList(str).push('mt')
        tab.children = [recursive_build(obj.globals.dtype, base, 'mt.globals'),
                        recursive_build(obj.row.dtype, base, 'mt.row'),
                        recursive_build(obj.col.dtype, base, 'mt.col'),
                        recursive_build(obj.entry.dtype, base, 'mt.entry')]

        tab.set_title(0, 'globals')
        tab.set_title(1, 'row')
        tab.set_title(2, 'col')
        tab.set_title(3, 'entry')
        tab.selected_index = 3
        selection_handler = widgets.IntText(3)

        box = widgets.VBox([widgets.HBox([glob, cols]), widgets.HBox([rows, entries])])
        buttons = [glob, rows, cols, entries]

    button_idx = dict(zip(buttons, range(len(buttons))))

    def handle_selection(x):
        if x['name'] == 'value' and x['type'] == 'change':
            buttons[x['old']].style = base_style
            selection = x['new']
            buttons[selection].style = selected_style
            tab.selected_index = selection

    selection_handler.observe(handle_selection)
    link((tab, 'selected_index'), (selection_handler, 'value'))

    def button_action(b):
        selection_handler.value = button_idx[b]

    for button in button_idx:
        button.on_click(button_action)

    display(box, tab)


def format_type(t):
    if isinstance(t, hl.tstruct):
        return 'struct'
    elif isinstance(t, hl.ttuple):
        return 'tuple'
    elif isinstance(t, hl.tarray):
        return 'array'
    elif isinstance(t, hl.tset):
        return 'set'
    elif isinstance(t, hl.tdict):
        return 'dict'
    else:
        return str(t)


def format_html(s):
    return '<p style="font-family:courier;white-space:pre;line-height: 115%;">{}</p>'.format(
        str(s).replace('<', '&lt').replace('>', '&gt').replace('\n', '</br>'))


def recursive_build(dtype, path=hl.utils.LinkedList(str), override_path=None):
    frames = []
    frames.append(widgets.Text(
        value=override_path if override_path is not None else ''.join(list(path)[::-1]),
        description='path: ',
        disabled=True,
        layout=Layout(width='auto')
    ))
    frames.append(widgets.Text(
        value=str(dtype),
        description='type: ',
        disabled=True,
        layout=Layout(width='auto')
    ))

    if isinstance(dtype, hl.tstruct):
        frames.append(widgets.HTML('<big>Fields:</big>'))
        acc = widgets.Accordion([recursive_build(v, path.push(f'[{repr(k)}]')) for k, v in dtype.items()])
        for i, (name, fd) in enumerate(dtype.items()):
            acc.set_title(i, f'[ {repr(name)} ]: {format_type(fd)}')
        acc.selected_index = None
        frames.append(acc)
    elif isinstance(dtype, hl.ttuple):
        frames.append(widgets.HTML('<big>Fields:</big>'))
        acc = widgets.Accordion([recursive_build(v, path.push(f'[{i}]')) for i, v in enumerate(dtype)])
        for i, fd in enumerate(dtype):
            acc.set_title(i, f'[ {i} ]: {format_type(fd)}')
        acc.selected_index = None
        frames.append(acc)
    elif isinstance(dtype, (hl.tarray, hl.tset)):
        acc = widgets.Accordion([recursive_build(dtype.element_type, path.push('[<element>]'))])
        acc.set_title(0, f'[ <element> ]: {format_type(dtype.element_type)}')
        acc.selected_index = None
        frames.append(acc)
    elif isinstance(dtype, (hl.tdict)):
        acc = widgets.Accordion([recursive_build(dtype.key_type, path.push('[<key>]')),
                                 recursive_build(dtype.value_type, path.push('[<value>]'))])
        acc.set_title(0, f'[ <key> ]: {format_type(dtype.key_type)}')
        acc.set_title(1, f'[ <value> ]: {format_type(dtype.value_type)}')
        acc.selected_index = None
        frames.append(acc)
    return widgets.VBox(frames)
