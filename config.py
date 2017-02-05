import json
import logging
import jsonschema
import yaml

from collections import UserDict


class Config(UserDict):
    """YAML File Configuration Manager"""

    def __init__(self, filename, schema):
        super().__init__()
        self._filename = filename
        try:
            with open(filename) as f:
                self.data = yaml.safe_load(f)
                with open(schema) as s:
                    self.schema = json.load(s)
                jsonschema.validate(self.data, self.schema)
        except FileNotFoundError:
            pass  # Just make an empty config; create on __exit__()
        except jsonschema.ValidationError as e:
            logging.warning("Your configuration is not valid: " + e.message)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        with open(self._filename, "w") as f:
            yaml.dump(self.data, f, indent=2, default_flow_style=False)

        return False  # propagate exceptions from the calling context

    def __getattr__(self, key):
        attr = getattr(super(Config, self), key, None)
        if attr:
            return attr
        # Keys contained dashes can be called using an underscore
        key = key.replace("_", "-")
        return self.data[key]
