import plotly
import hail as hl


def check_scale_continuity(scale, dtype, aes_key):

    if scale.is_discrete() and not is_discrete_type(dtype):
        raise ValueError(f"Aesthetic {aes_key} has discrete scale but not a discrete type.")
    if scale.is_continuous() and not is_continuous_type(dtype):
        raise ValueError(f"Aesthetic {aes_key} has continuous scale but not a continuous type.")


def is_genomic_type(dtype):
    return isinstance(dtype, hl.tlocus)


def is_continuous_type(dtype):
    return dtype in [hl.tint32, hl.tint64, hl.tfloat32, hl.tfloat64]


def is_discrete_type(dtype):
    return dtype in [hl.tstr]


excluded_from_grouping = {"x", "tooltip", "label"}


def should_use_for_grouping(name, type):
    return (name not in excluded_from_grouping) and is_discrete_type(type)


def should_use_scale_for_grouping(scale):
    return (scale.aesthetic_name not in excluded_from_grouping) and scale.is_discrete()


# Map strings to numbers that will index into a color scale.
def categorical_strings_to_colors(string_set, color_values):

    if isinstance(color_values, list):
        if len(string_set) > len(color_values):
            print(f"Not enough colors specified. Found {len(string_set)} distinct values of color aesthetic and only {len(color_values)} colors were provided.")
        color_dict = {}
        for idx, element in enumerate(string_set):
            if element not in color_dict:
                color_dict[element] = color_values[idx]

    else:
        color_dict = color_values

    return color_dict


def continuous_nums_to_colors(min_color, max_color, continuous_color_scale):
    def adjust_color(input_color):
        return (input_color - min_color) / max_color - min_color

    def transform_color(input_color):
        return plotly.colors.sample_colorscale(continuous_color_scale, adjust_color(input_color))[0]
    return transform_color


def bar_position_plotly_to_gg(plotly_pos):
    ggplot_to_plotly = {'dodge': 'group', 'stack': 'stack', 'identity': 'overlay'}
    return ggplot_to_plotly[plotly_pos]


def linetype_plotly_to_gg(plotly_linetype):
    linetype_dict = {
        "solid": "solid",
        "dashed": "dash",
        "dotted": "dot",
        "longdash": "longdash",
        "dotdash": "dashdot"
    }
    return linetype_dict[plotly_linetype]
