import json
import os


def get_cache_path(cache_name):
    return os.path.join("cache", cache_name + ".json")


def get_cache(cache_name, force=False):
    cache_path = get_cache_path(cache_name)
    if force or not os.path.isfile(cache_path):
        return {}
    with open(cache_path) as fl:
        return json.load(fl)


def save_cache(cache, cache_name):
    cache_path = get_cache_path(cache_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json_string = json.dumps(cache)
    with open(cache_path, "w") as fl:
        return fl.write(json_string)
