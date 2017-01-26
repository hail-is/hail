{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoattribute:: {{ objname }}
    :annotation:

.. autoclass:: {{ objname }}

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

    .. automethod:: __init__


