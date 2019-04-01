from datetime import date, datetime
import re
from kubernetes.client import models


def deserialize(api_client, data, klass):  # pylint: disable=R0911
    if data is None:
        return None

    if isinstance(klass, str):
        if klass.startswith('list['):
            sub_kls = re.match(r'list\[(.*)\]', klass).group(1)
            return [deserialize(api_client, sub_data, sub_kls)
                    for sub_data in data]

        if klass.startswith('dict('):
            sub_kls = re.match(r'dict\(([^,]*), (.*)\)', klass).group(2)
            return {k: deserialize(api_client, v, sub_kls)
                    for k, v in data.items()}

        # convert str to class
        if klass in api_client.NATIVE_TYPES_MAPPING:
            klass = api_client.NATIVE_TYPES_MAPPING[klass]
        else:
            klass = getattr(models, klass)

    if klass in api_client.PRIMITIVE_TYPES:
        return deserialize_primitive(data, klass)
    if klass == object:
        return deserialize_object(data)
    if klass == date:
        return deserialize_date(data)
    if klass == datetime:
        return deserialize_datatime(data)
    else:
        return deserialize_model(api_client, data, klass)


def deserialize_primitive(data, klass):
    try:
        return klass(data)
    # except UnicodeEncodeError:
    #     return unicode(data)  # FIXME: What should this be?
    except TypeError:
        return data


def deserialize_date(string):
    try:
        from dateutil.parser import parse
        return parse(string).date()
    except ImportError:
        return string
    except ValueError:
        raise Exception("Failed to parse `{0}` into a date object".format(string))


def deserialize_datatime(string):
    try:
        from dateutil.parser import parse
        return parse(string)
    except ImportError:
        return string
    except ValueError:
        raise Exception("Failed to parse `{0}` into a datetime object")


def deserialize_object(value):
    return value


def deserialize_model(api_client, data, klass):
    if not klass.swagger_types and not hasattr(klass, 'get_real_child_model'):
        return data

    kwargs = {}
    if klass.swagger_types is not None:
        for attr, attr_type in klass.swagger_types.items():
            if data is not None \
                    and attr in data \
                    and isinstance(data, (list, dict)):
                value = data[attr]
                kwargs[attr] = deserialize(api_client, value, attr_type)

    instance = klass(**kwargs)

    if hasattr(instance, 'get_real_child_model'):
        klass_name = instance.get_real_child_model(data)
        if klass_name:
            instance = deserialize(api_client, data, klass_name)
    return instance
