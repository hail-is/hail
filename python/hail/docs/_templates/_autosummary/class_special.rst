{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :special-members:
    :exclude-members: __dict__, __weakref__, __module__
    :no-undoc-members:

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
        :nosignatures:

    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% if methods %}

    .. rubric:: Methods

    .. autosummary::
        :nosignatures:

    {% for item in methods %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}