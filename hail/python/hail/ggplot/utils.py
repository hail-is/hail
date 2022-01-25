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


# Map strings to numbers that will index into a color scale.
def categorical_strings_to_colors(string_set, parent_plot):

    color_dict = parent_plot.discrete_color_dict

    for element in string_set:
        if element not in color_dict:
            color_dict[element] = parent_plot.discrete_color_scale[parent_plot.discrete_color_idx % len(parent_plot.discrete_color_scale)]
            parent_plot.discrete_color_idx += 1

    return parent_plot.discrete_color_dict


def continuous_nums_to_colors(input_color_nums, continuous_color_scale):
    min_color = min(input_color_nums)
    max_color = max(input_color_nums)

    def adjust_color(input_color):
        return (input_color - min_color) / max_color - min_color

    color_mapping = plotly.colors.sample_colorscale(continuous_color_scale, [adjust_color(input_color) for input_color in input_color_nums])
    return color_mapping
