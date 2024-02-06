class Box:
    def __init__(
        self,
        horizontal: str,
        top_horizontal_split: str,
        bottom_horizontal_split: str,
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
        self.ts = self.top_split = self.top_horizontal_split = top_horizontal_split
        self.bs = self.bottom_split = self.bottom_horizontal_split = bottom_horizontal_split
        self.l = self.left = self.r = self.right = self.v = self.vertical = vertical
        self.ls = self.left_split = self.left_vertical_split = left_vertical_split
        self.rs = self.right_split = self.right_vertical_split = right_vertical_split

        self.p = self.plus = plus

        self.tl = self.top_left = top_left
        self.tr = self.top_right = top_right
        self.br = self.bottom_right = bottom_right
        self.bl = self.bottom_left = bottom_left


standard = Box('─', '┬', '┴', '│', '├', '┤', '┼', '┌', '┐', '┘', '└')
thick = Box('━', '┳', '┻', '┃', '┣', '┫', '╋', '┏', '┓', '┛', '┗')
double = Box('═', '╦', '╩', '║', '╠', '╣', '╬', '╔', '╗', '╝', '╚')

two_dashed_standard = Box('╌', '┬', '┴', '╎', '├', '┤', '┼', '┌', '┐', '┘', '└')
three_dashed_standard = Box('┄', '┬', '┴', '┆', '├', '┤', '┼', '┌', '┐', '┘', '└')
four_dashed_standard = Box('┈', '┬', '┴', '┊', '├', '┤', '┼', '┌', '┐', '┘', '└')

curved = Box('─', '┬', '┴', '│', '├', '┤', '┼', '╭', '╮', '╯', '╰')
two_dashed_curved = Box('╌', '┬', '┴', '╎', '├', '┤', '┼', '╭', '╮', '╯', '╰')
three_dashed_curved = Box('┄', '┬', '┴', '┆', '├', '┤', '┼', '╭', '╮', '╯', '╰')
four_dashed_curved = Box('┈', '┬', '┴', '┊', '├', '┤', '┼', '╭', '╮', '╯', '╰')

two_dashed_thick = Box('╍', '┳', '┻', '╏', '┣', '┫', '╋', '┏', '┓', '┛', '┗')
three_dashed_thick = Box('┅', '┳', '┻', '┇', '┣', '┫', '╋', '┏', '┓', '┛', '┗')
four_dashed_thick = Box('┉', '┳', '┻', '┋', '┣', '┫', '╋', '┏', '┓', '┛', '┗')

# The "Box Drawing" block also supports mixed weight and mixed
# single-double. https://en.wikipedia.org/wiki/Box-drawing_character
