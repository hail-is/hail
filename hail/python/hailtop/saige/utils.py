from typing import Dict, Optional


def rectify_name(config_name: Optional[str], input_name: Optional[str]) -> Optional[str]:
    return input_name or config_name


def rectify_attributes(config_attrs: Optional[Dict[str, str]], input_attrs: Optional[Dict[str, str]]) -> Dict[str, str]:
    attributes = {}
    if config_attrs:
        attributes.update(config_attrs)
    if input_attrs:
        attributes.update(input_attrs)
    return attributes
