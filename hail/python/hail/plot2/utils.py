# Map strings to numbers that will index into a color scale.
def categorical_strings_to_colors(string_set):
    color_dict={}
    x = 0

    for element in string_set:
        if element not in color_dict:
            color_dict[element] = x
            x += 1

    return color_dict