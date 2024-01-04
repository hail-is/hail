from typing import Union, Dict, Pattern, Callable, Any, List, Optional
import re
import logging

log = logging.getLogger('foo')


class ValidationError(Exception):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason


class RequiredKey:
    def __init__(self, key: str):
        self.key = key


Key = Union[str, RequiredKey]


class TypedValidator:
    def __init__(self, t):
        self.t = t

    def validate(self, name, obj):
        if not isinstance(obj, self.t):
            raise ValidationError(f'{name} is not {self.t}')


class DictValidator(TypedValidator):
    def __init__(self, vchecker: 'Validator'):
        super().__init__(dict)
        self.vchecker = vchecker

    def validate(self, name: str, obj):
        super().validate(name, obj)
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValidationError(f'{name} has non-str key')
            self.vchecker.validate(f"{name}[{k}]", v)


class KeyedValidator(TypedValidator):
    def __init__(self, keyed_checkers: Dict[Key, 'Validator']):
        super().__init__(dict)
        self.checkers = {}
        for k, v in keyed_checkers.items():
            if isinstance(k, RequiredKey):
                self.checkers[k.key] = (v, True)
            else:
                self.checkers[k] = (v, False)

    def __getitem__(self, key: str):
        return self.checkers[key][0]

    def validate(self, name: str, obj):
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
    def __init__(self, checker: 'Validator'):
        super().__init__(list)
        self.checker = checker

    def validate(self, name: str, obj):
        super().validate(name, obj)
        for i, elt in enumerate(obj):
            self.checker.validate(f"{name}[{i}]", elt)


class SetValidator:
    def __init__(self, valid):
        self.valid = valid

    def validate(self, name: str, obj):
        if obj not in self.valid:
            raise ValidationError(f'{name} must be one of: {self.valid}')


class RegexValidator(TypedValidator):
    def __init__(self, pattern: str, re_obj: Optional[Pattern], maxlen: Optional[int]):
        super().__init__(str)
        self.pattern = pattern
        self.re_obj = re_obj if re_obj is not None else re.compile(pattern)
        self.maxlen = maxlen

    def validate(self, name: str, obj):
        super().validate(name, obj)
        if self.maxlen is not None and len(obj) > self.maxlen:
            raise ValidationError(f'length of {name} must be <= {self.maxlen}')
        if not self.re_obj.fullmatch(obj):
            raise ValidationError(f'{name} must match regex: {self.pattern}')


class NumericValidator:
    def __init__(self, conditions: Dict[str, Callable[[Any], Any]]):
        self.conditions = conditions

    def validate(self, name: str, obj):
        if not isinstance(obj, int) and not isinstance(obj, float):
            raise ValidationError(f'{name} is not numeric')
        for text, condition in self.conditions.items():
            if not condition(obj):
                raise ValidationError(f'{name} does not satisfy the condition {text}')


class SwitchValidator(TypedValidator):
    def __init__(self, key: str, checkers: Dict[str, Dict[Key, 'Validator']]):
        super().__init__(dict)
        self.key = key
        self.valid_key = oneof(*checkers.keys())
        self.checkers = {k: keyed({required(key): self.valid_key, **fields}) for k, fields in checkers.items()}

    def __getitem__(self, key):
        return self.checkers[key]

    def validate(self, name: str, obj):
        super().validate(name, obj)
        key = obj[self.key]
        self.valid_key.validate(f"{name}.{key}", key)
        self.checkers[key].validate(f"{name}", obj)


class NullableValidator:
    def __init__(self, wrapped: 'Validator'):
        self.checker = wrapped

    def validate(self, name: str, obj):
        if obj is not None:
            self.checker.validate(name, obj)


class TruthyValidator:
    def validate(self, name: str, obj):
        if not obj:
            raise ValidationError(f'{name} cannot be {obj}')


class MultipleValidator:
    def __init__(self, checkers: List['Validator']):
        self.checkers = checkers

    def validate(self, name: str, obj):
        excs = []
        for checker in self.checkers:
            try:
                checker.validate(name, obj)
                return
            except ValidationError as e:
                excs.append(e)
        if excs:
            reasons = ' or '.join([e.reason for e in excs])
            log.info(reasons)
            raise ValidationError(f'{name} does not satisfy any conditions: {reasons}')


def required(key: str) -> RequiredKey:
    return RequiredKey(key)


str_type = TypedValidator(str)
non_empty_str_type = MultipleValidator([str_type, TruthyValidator()])
bool_type = TypedValidator(bool)
int_type = TypedValidator(int)

Validator = Union[TypedValidator, NumericValidator, NullableValidator, TruthyValidator, SetValidator, MultipleValidator]


def dictof(vchecker: Validator):
    return DictValidator(vchecker)


def keyed(checkers: Dict[Key, Validator]):
    return KeyedValidator(checkers)


def listof(checker: Validator):
    return ListValidator(checker)


def oneof(*items):
    return SetValidator(set(items))


def regex(pattern, re_obj=None, maxlen=None):
    return RegexValidator(pattern, re_obj, maxlen)


def nullable(wrapped: Validator):
    return NullableValidator(wrapped)


def numeric(**conditions: Callable[[Any], Any]):
    return NumericValidator(conditions)


def switch(key: str, checkers: Dict[str, Dict[Key, Validator]]):
    return SwitchValidator(key, checkers)


def anyof(*checkers: Validator):
    return MultipleValidator(list(checkers))
