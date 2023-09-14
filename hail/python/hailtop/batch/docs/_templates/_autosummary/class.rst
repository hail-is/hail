{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}()
    :show-inheritance:
    :members:
    :no-inherited-members:

    {% block attributes %}
    {% if (attributes | reject('in', inherited_members) | list | count) != 0 %}
    .. rubric:: Attributes

    .. autosummary::
        :nosignatures:

    {% for item in attributes %}
    {% if item not in inherited_members %}
        ~{{ name }}.{{ item }}
    {% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% if (methods | reject('in', inherited_members) | list | count) != 0 %}

    .. rubric:: Methods

    .. autosummary::
        :nosignatures:

    {% for item in methods %}
    {% if item not in inherited_members %}
        ~{{ name }}.{{ item }}
    {% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}
