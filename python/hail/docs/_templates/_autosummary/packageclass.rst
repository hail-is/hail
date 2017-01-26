{{ fullname }}
{{ underline }}


.. automodule:: {{ fullname }}

   .. rubric:: Classes

   .. toctree::
   .. autosummary::
        :nosignatures:

{% for item in classes %}
        .. automethod:: __init__
            .. autosummary::
            :nosignatures:
            :toctree: ./{{objname}}
            :template: class.rst

            {{ item }}
        {%- endfor %}
