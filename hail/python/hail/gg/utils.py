import plotly

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
    adjust_color = lambda input_color: (input_color - min_color) / max_color - min_color
    color_mapping = plotly.colors.sample_colorscale(continuous_color_scale, [adjust_color(input_color) for input_color in input_color_nums])
    return color_mapping