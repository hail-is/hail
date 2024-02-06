from typing import List


class Box:
    t: str
    top: str
    b: str
    bottom: str
    h: str
    horizontal: str
    ts: str
    top_split: str
    bs: str
    bottom_split: str
    l: str
    left: str
    r: str
    right: str
    v: str
    vertical: str
    ls: str
    left_split: str
    left_vertical_split: str
    rs: str
    right_split: str
    right_vertical_split: str
    p: str
    tl: str
    top_left: str
    tr: str
    top_right: str
    br: str
    bottom_right: str
    bl: str
    bottom_left: str


class SimpleBox(Box):
    def __init__(
        self,
        horizontal: str,
        top_split: str,
        bottom_split: str,
        vertical: str,
        left_vertical_split: str,
        right_vertical_split: str,
        plus: str,
        top_left: str,
        top_right: str,
        bottom_right: str,
        bottom_left: str,
    ):
        self.t = self.top = self.b = self.bottom = self.h = self.horizontal = horizontal
        self.ts = self.top_split = top_split
        self.bs = self.bottom_split = bottom_split
        self.l = self.left = self.r = self.right = self.v = self.vertical = vertical
        self.ls = self.left_split = self.left_vertical_split = left_vertical_split
        self.rs = self.right_split = self.right_vertical_split = right_vertical_split

        self.p = self.plus = plus

        self.tl = self.top_left = top_left
        self.tr = self.top_right = top_right
        self.br = self.bottom_right = bottom_right
        self.bl = self.bottom_left = bottom_left


standard = SimpleBox('─', '┬', '┴', '│', '├', '┤', '┼', '┌', '┐', '┘', '└')
thick = SimpleBox('━', '┳', '┻', '┃', '┣', '┫', '╋', '┏', '┓', '┛', '┗')
double = SimpleBox('═', '╦', '╩', '║', '╠', '╣', '╬', '╔', '╗', '╝', '╚')

two_dashed_standard = SimpleBox('╌', '┬', '┴', '╎', '├', '┤', '┼', '┌', '┐', '┘', '└')
three_dashed_standard = SimpleBox('┄', '┬', '┴', '┆', '├', '┤', '┼', '┌', '┐', '┘', '└')
four_dashed_standard = SimpleBox('┈', '┬', '┴', '┊', '├', '┤', '┼', '┌', '┐', '┘', '└')

curved = SimpleBox('─', '┬', '┴', '│', '├', '┤', '┼', '╭', '╮', '╯', '╰')
two_dashed_curved = SimpleBox('╌', '┬', '┴', '╎', '├', '┤', '┼', '╭', '╮', '╯', '╰')
three_dashed_curved = SimpleBox('┄', '┬', '┴', '┆', '├', '┤', '┼', '╭', '╮', '╯', '╰')
four_dashed_curved = SimpleBox('┈', '┬', '┴', '┊', '├', '┤', '┼', '╭', '╮', '╯', '╰')

two_dashed_thick = SimpleBox('╍', '┳', '┻', '╏', '┣', '┫', '╋', '┏', '┓', '┛', '┗')
three_dashed_thick = SimpleBox('┅', '┳', '┻', '┇', '┣', '┫', '╋', '┏', '┓', '┛', '┗')
four_dashed_thick = SimpleBox('┉', '┳', '┻', '┋', '┣', '┫', '╋', '┏', '┓', '┛', '┗')

# The "Box Drawing" block also supports mixed weight and mixed
# single-double. https://en.wikipedia.org/wiki/Box-drawing_character


class GeneralBox(Box):
    def __init__(
        self,
        top: str,
        horizontal: str,
        bottom: str,
        top_split: str,
        bottom_split: str,
        left: str,
        right: str,
        vertical: str,
        left_split: str,
        right_split: str,
        plus: str,
        top_left: str,
        top_right: str,
        bottom_right: str,
        bottom_left: str,
    ):
        self.t = self.top = top
        self.h = self.horizontal = horizontal
        self.b = self.bottom = bottom

        self.ts = self.top_split = top_split
        self.bs = self.bottom_split = bottom_split

        self.l = self.left = left
        self.r = self.right = right
        self.v = self.vertical = vertical
        self.ls = self.left_split = left_split
        self.rs = self.right_split = right_split

        self.p = self.plus = plus

        self.tl = self.top_left = top_left
        self.tr = self.top_right = top_right
        self.br = self.bottom_right = bottom_right
        self.bl = self.bottom_left = bottom_left


standard_thick = GeneralBox('─', '─', '━', '┬', '┻', '│', '│', '│', '┢', '┪', '╈', '┌', '┐', '┙', '┕')
curved_thick = GeneralBox('─', '─', '━', '┬', '┻', '│', '│', '│', '┢', '┪', '╈', '╭', '╮', '┙', '┕')
thick_standard = GeneralBox('━', '━', '━', '┳', '┴', '┃', '┃', '┃', '┡', '┩', '╇', '┏', '┓', '┚', '┖')

# U+250x	─	━	│	┃	┄	┅	┆	┇	┈	┉	┊	┋	┌	┍	┎	┏
# U+251x	┐	┑	┒	┓	└	┕	┖	┗	┘	┙	┚	┛	├	┝	┞	┟
# U+252x	┠	┡	┢	┣	┤	┥	┦	┧	┨	┩	┪	┫	┬	┭	┮	┯
# U+253x	┰	┱	┲	┳	┴	┵	┶	┷	┸	┹	┺	┻	┼	┽	┾	┿
# U+254x	╀	╁	╂	╃	╄	╅	╆	╇	╈	╉	╊	╋	╌	╍	╎	╏
# U+255x	═	║	╒	╓	╔	╕	╖	╗	╘	╙	╚	╛	╜	╝	╞	╟
# U+256x	╠	╡	╢	╣	╤	╥	╦	╧	╨	╩	╪	╫	╬	╭	╮	╯


class HLine:
    def __init__(
        self,
        left: str,
        horizontal: str,
        horizontal_split: str,
        right: str,
    ):
        self.l = self.left = left
        self.h = self.horizontal = horizontal
        self.hs = self.horizontal_split = horizontal_split
        self.r = self.right = right


standard_top_hline = HLine('┌', '─', '┬', '┐')
standard_mid_hline = HLine('├', '─', '┼', '┤')
standard_box_hline = HLine('│', '─', '│', '│')
standard_bot_hline = HLine('└', '─', '┴', '┘')

curved_top_hline = HLine('╭', '─', '┬', '╮')
curved_mid_hline = HLine('├', '─', '┼', '┤')
curved_box_hline = HLine('│', '─', '│', '│')
curved_bot_hline = HLine('╰', '─', '┴', '╯')

thick_top_hline = HLine('┏', '━', '┳', '┓')
thick_mid_hline = HLine('┣', '━', '╋', '┫')
thick_box_hline = HLine('┃', '━', '┃', '┃')
thick_bot_hline = HLine('┗', '━', '┻', '┛')

double_top_hline = HLine('╔', '═', '╦', '╗')
double_mid_hline = HLine('╠', '═', '╬', '╣')
double_box_hline = HLine('║', '═', '║', '║')
double_bot_hline = HLine('╚', '═', '╩', '╝')

standard_thick_top_hline = HLine('┎', '─', '┰', '┒')
standard_thick_mid_hline = HLine('┟', '─', '╁', '┧')
standard_thick_box_hline = HLine('╽', '─', '╽', '╽')
standard_thick_bot_hline = HLine('┖', '─', '┸', '┚')

thick_standard_top_hline = HLine('┍', '━', '┯', '┑')
thick_standard_mid_hline = HLine('┡', '━', '╁', '┩')
thick_standard_box_hline = HLine('╽', '━', '╽', '╽')
thick_standard_bot_hline = HLine('┖', '━', '┸', '┚')

two_dashed_standard_top_hline = HLine('┌', '╌', '┬', '┐')
two_dashed_standard_mid_hline = HLine('├', '╌', '┼', '┤')
two_dashed_standard_box_hline = HLine('╎', '╌', '╎', '╎')
two_dashed_standard_bot_hline = HLine('└', '╌', '┴', '┘')

three_dashed_standard_top_hline = HLine('┌', '┄', '┬', '┐')
three_dashed_standard_mid_hline = HLine('├', '┄', '┼', '┤')
three_dashed_standard_box_hline = HLine('┆', '┄', '┆', '┆')
three_dashed_standard_bot_hline = HLine('└', '┄', '┴', '┘')

four_dashed_standard_top_hline = HLine('┌', '┈', '┬', '┐')
four_dashed_standard_mid_hline = HLine('├', '┈', '┼', '┤')
four_dashed_standard_box_hline = HLine('┊', '┈', '┊', '┊')
four_dashed_standard_bot_hline = HLine('└', '┈', '┴', '┘')

two_dashed_thick_top_hline = HLine('┏', '╍', '┳', '┓')
two_dashed_thick_mid_hline = HLine('┣', '╍', '╋', '┫')
two_dashed_thick_box_hline = HLine('╏', '╍', '╏', '╏')
two_dashed_thick_bot_hline = HLine('┗', '╍', '┻', '┛')

three_dashed_thick_top_hline = HLine('┏', '┅', '┳', '┓')
three_dashed_thick_mid_hline = HLine('┣', '┅', '╋', '┫')
three_dashed_thick_box_hline = HLine('┇', '┅', '┇', '┇')
three_dashed_thick_bot_hline = HLine('┗', '┅', '┻', '┛')

four_dashed_thick_top_hline = HLine('┏', '┉', '┳', '┓')
four_dashed_thick_mid_hline = HLine('┣', '┉', '╋', '┫')
four_dashed_thick_box_hline = HLine('┋', '┉', '┋', '┋')
four_dashed_thick_bot_hline = HLine('┗', '┉', '┻', '┛')


class TableStyle:
    def __init__(
        self,
        header_box: Box,
        transition_box: Box,
        body_box: Box,
        border_top: bool,
        border_bot: bool,
        border_left: bool,
        border_right: bool,
    ):
        self.hb = self.header_box = header_box
        self.tb = self.transition_box = transition_box
        self.bb = self.body_box = body_box
        self.border_top = border_top
        self.border_bot = border_bot
        self.border_left = border_left
        self.border_right = border_right

    def format_line(self, widths: List[str], kind: str) -> str:
        if kind == 'top':
            if not self.border_top:
                return ''

            empty = ''
            if self.border_left:
                empty += self.hb.tl
            if self.border_right:
                empty += self.hb.tr

            left = self.hb.tl + self.hb.t if self.border_left else self.hb.t
            splitter = self.hb.t + self.hb.ts + self.hb.t
            right = self.hb.t + self.hb.tr if self.border_right else self.hb.t
        elif kind == 'top-inner':
            empty = ''
            if self.border_left:
                empty += self.hb.ls
            if self.border_right:
                empty += self.hb.rs

            left = self.hb.ls + self.hb.h if self.border_left else self.hb.h
            splitter = self.hb.h + self.hb.p + self.hb.h
            right = self.hb.h + self.hb.rs if self.border_right else self.hb.h
        elif kind == 'top-bottom':
            empty = ''
            if self.border_left:
                empty += self.tb.ls
            if self.border_right:
                empty += self.tb.rs

            left = self.tb.ls + self.tb.b if self.border_left else self.tb.b
            splitter = self.tb.b + self.tb.p + self.tb.b
            right = self.tb.b + self.tb.rs if self.border_right else self.tb.b
        elif kind == 'bottom-inner':
            empty = ''
            if self.border_left:
                empty += self.bb.ls
            if self.border_right:
                empty += self.bb.rs

            left = self.bb.ls + self.bb.h if self.border_left else self.bb.h
            splitter = self.bb.h + self.bb.p + self.bb.h
            right = self.bb.h + self.bb.rs if self.border_right else self.bb.h
        else:
            assert kind == 'bottom'
            if not self.border_bot:
                return ''

            empty = ''
            if self.border_left:
                empty += self.bb.bl
            if self.border_right:
                empty += self.bb.br

            left = self.bb.bl + self.bb.b if self.border_left else self.bb.b
            splitter = self.bb.b + self.bb.bs + self.bb.b
            right = self.bb.b + self.bb.br if self.border_right else self.bb.b

        if not widths:
            return empty + '\n'

        if kind in ('top', 'top-inner'):
            line_type = self.hb.b
        elif kind == 'top-bottom':
            line_type = self.tb.b
        else:
            assert kind in ('bottom-inner', 'bottom')
            line_type = self.bb.b

        contents = splitter.join([line_type * width for width in widths])
        return left + contents + right + '\n'

    def format_row(self, values: List[str], *, header: bool = False):
        box = self.hb if header else self.bb
        if not values:
            s = ''
            if self.border_left:
                s += box.l
            if self.border_right:
                s += box.r
            return s + '\n'
        left_border = box.l + ' ' if self.border_left else ''
        contents = (' ' + box.v + ' ').join(values)
        right_border = ' ' + box.r if self.border_right else ''
        return left_border + contents + right_border + '\n'


standard_ts = TableStyle(thick, thick_standard, standard, True, True, True, True)
# markdown_ts = TableStyle(thick_standard, standard, True, True, True, True)
