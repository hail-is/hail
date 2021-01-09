import re

class ValidationError(Exception):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason


class RequiredKey:
    def __init__(self, key):
        self.key = key


class TypedValidator:
    def __init__(self, t):
        self.t = t

    def validate(self, name, obj):
        if not isinstance(obj, self.t):
            raise ValidationError(f'{name} is not {self.t}')


class DictValidator(TypedValidator):
    def __init__(self, vchecker):
        super().__init__(dict)
        self.vchecker = vchecker

    def validate(self, name, obj):
        super().validate(name, obj)
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValidationError(f'{name} has non-str key')
            self.vchecker.validate(f"{name}[{k}]", v)


class KeyedValidator(TypedValidator):
    def __init__(self, keyed_checkers):
        super().__init__(dict)
        self.checkers = {}
        for k, v in keyed_checkers.items():
            if isinstance(k, RequiredKey):
                self.checkers[k.key] = (v, True)
            else:
                self.checkers[k] = (v, False)

    def __getitem__(self, key):
        return self.checkers[key][0]

    def validate(self, name, obj):
        super().validate(name, obj)
        unknown_keys = set(obj.keys()) - set(self.checkers.keys())
        if len(unknown_keys) != 0:
            raise ValidationError(f'unknown keys in {name}: {unknown_keys}')
        for k, (checker, required) in self.checkers.items():
            if required and k not in obj:
                raise ValidationError(f'{name} missing required key {k}.')
            if k in obj:
                checker.validate(f"{name}.{k}", obj[k])


class ListValidator(TypedValidator):
    def __init__(self, checker):
        super().__init__(list)
        self.checker = checker

    def validate(self, name, obj):
        super().validate(name, obj)
        for i, elt in enumerate(obj):
            self.checker.validate(f"{name}[{i}]", elt)


class SetValidator:
    def __init__(self, valid):
        self.valid = valid

    def validate(self, name, obj):
        if obj not in self.valid:
            raise ValidationError(f'{name} must be one of: {self.valid}')


class RegexValidator(TypedValidator):
    def __init__(self, pattern, re_obj, maxlen):
        super().__init__(str)
        self.pattern = pattern
        self.re_obj = re_obj if re_obj is not None else re.compile(pattern)
        self.maxlen = maxlen

    def validate(self, name, obj):
        super().validate(name, obj)
        if self.maxlen is not None and len(obj) > self.maxlen:
            raise ValidationError(f'length of {name} must be <= {self.maxlen}')
        if not self.re_obj.fullmatch(obj):
            raise ValidationError(f'{name} must match regex: {self.pattern}')


class NumericValidator:
    def __init__(self, conditions):
        self.conditions = conditions

    def validate(self, name, obj):
        if not isinstance(obj, int) and not isinstance(obj, float):
            raise ValidationError(f'{name} is not numeric')
        for text, condition in self.conditions.items():
            if not condition(obj):
                raise ValidationError(f'{name} does not satisfy the condition {text}')


class SwitchValidator(TypedValidator):
    def __init__(self, key, checkers):
        super().__init__(dict)
        self.key = key
        self.valid_key = oneof(*checkers.keys())
        self.checkers = {k: keyed({required(key): self.valid_key, **fields}) for k, fields in checkers.items()}

    def __getitem__(self, key):
        return self.checkers[key]

    def validate(self, name, obj):
        super().validate(name, obj)
        key = obj[self.key]
        self.valid_key.validate(f"{name}.{key}", key)
        self.checkers[key].validate(obj)


class NullableValidator:
    def __init__(self, wrapped):
        self.checker = wrapped

    def validate(self, name, obj):
        if obj is not None:
            self.checker.validate(name, obj)


def required(key):
    return RequiredKey(key)

str_type = TypedValidator(str)
bool_type = TypedValidator(bool)
int_type = TypedValidator(int)


def dictof(vchecker):
    return DictValidator(vchecker)


def keyed(checkers):
    return KeyedValidator(checkers)


def listof(checker):
    return ListValidator(checker)


def oneof(*items):
    return SetValidator(set(items))


def regex(pattern, re_obj=None, maxlen=None):
    return RegexValidator(pattern, re_obj, maxlen)


def nullable(wrapped):
    return NullableValidator(wrapped)


def numeric(**conditions):
    return NumericValidator(conditions)


def switch(key, checkers):
    return SwitchValidator(key, checkers)
