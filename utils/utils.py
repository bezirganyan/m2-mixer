from typing import Dict, Any, TypeVar, Union

from omegaconf import DictConfig

KeyType = TypeVar('KeyType')


def deep_update(mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], (dict, DictConfig)) and isinstance(v, (
            dict, DictConfig)):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def todict(obj: Union[Dict, DictConfig]) -> Dict:
    if isinstance(obj, DictConfig):
        obj = dict(obj)
    if isinstance(obj, (dict, DictConfig)):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v)
        return data
    else:
        return obj
