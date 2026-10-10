"""Prompts are versioned artifacts: prompts/<prompt_id>/<version>.yaml.

The active version per prompt is chosen by PROMPT_VERSION_<ID> (default v1). Every LLM call and every run result
records `<id>@<version>#<sha8>`, where sha8 is the hash of the template text, so an edited file without a version
bump is still distinguishable in the logs.
"""
from __future__ import annotations

import hashlib
import os
import string
from dataclasses import dataclass
from functools import lru_cache

import yaml


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    version: str
    system: str
    user: str
    description: str
    sha8: str

    @property
    def tag(self) -> str:
        return f"{self.prompt_id}@{self.version}#{self.sha8}"

    def render(self, **values) -> list[dict[str, str]]:
        fmt = _SafeFormatter()
        return [{"role": "system", "content": fmt.format(self.system, **values)},
                {"role": "user", "content": fmt.format(self.user, **values)}]


class _SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        return kwargs.get(key, "") if isinstance(key, str) else super().get_value(key, args, kwargs)


@lru_cache(maxsize=64)
def load_prompt(prompts_dir: str, prompt_id: str, version: str) -> Prompt:
    path = os.path.join(prompts_dir, prompt_id, f"{version}.yaml")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    text = raw["system"] + "\n---\n" + raw["user"]
    return Prompt(prompt_id=prompt_id, version=version, system=raw["system"], user=raw["user"],
                  description=raw.get("description", ""), sha8=hashlib.sha256(text.encode()).hexdigest()[:8])


def available_versions(prompts_dir: str, prompt_id: str) -> list[str]:
    d = os.path.join(prompts_dir, prompt_id)
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))
