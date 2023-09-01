import dataclasses


def cast_dataclass_attributes(dc):
    for f in dataclasses.fields(dc):
        value = getattr(dc, f.name)
        if not isinstance(value, f.type):
            if isinstance(f.type, bool):
                if value in ('0', 'false', 'False'):
                    value = False
                elif value in ('1', 'true', 'True'):
                    value = True
            setattr(dc, f.name, f.type(value))
