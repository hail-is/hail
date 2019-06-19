from ..utils.java import Env


def compile_comparison_binary(op, codecName, l_type, r_type):
    return Env.hc()._jhc.backend().compileComparisonBinary(
        op, codecName, l_type, r_type)
