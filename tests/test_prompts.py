"""Versioned prompt artifacts (triage.prompts) and the files under prompts/."""
import os

import pytest
import yaml

from triage.prompts import Prompt, available_versions, load_prompt

PROMPTS = os.environ["PROMPTS_DIR"]


def test_tag_and_hash_are_stable():
    a = load_prompt(PROMPTS, "planner", "v1")
    b = load_prompt.__wrapped__(PROMPTS, "planner", "v1")   # bypass the cache: same file, same hash
    assert a.sha8 == b.sha8 and len(a.sha8) == 8
    assert a.tag == f"planner@v1#{a.sha8}"


def test_different_versions_have_different_hashes():
    v1 = load_prompt(PROMPTS, "extraction", "v1")
    v2 = load_prompt(PROMPTS, "extraction", "v2")
    assert v1.sha8 != v2.sha8 and v1.tag != v2.tag


def test_render_fills_missing_values_with_blank():
    p = Prompt("x", "v1", "System for {who}.", "Facts: {facts}; extra: {missing}", "", "abcd1234")
    msgs = p.render(who="officer", facts="{a: 1}")
    assert msgs == [{"role": "system", "content": "System for officer."},
                    {"role": "user", "content": "Facts: {a: 1}; extra: "}]


def test_missing_version_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt(PROMPTS, "planner", "v999")


@pytest.mark.parametrize("prompt_id", sorted(os.listdir(PROMPTS)))
def test_every_prompt_file_is_well_formed(prompt_id):
    versions = available_versions(PROMPTS, prompt_id)
    assert "v1" in versions, f"{prompt_id} has no v1"
    for v in versions:
        with open(os.path.join(PROMPTS, prompt_id, f"{v}.yaml"), encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        assert raw.get("system", "").strip() and raw.get("user", "").strip(), f"{prompt_id}/{v} is missing text"
        if "id" in raw:
            assert raw["id"] == prompt_id
        if "version" in raw:
            assert raw["version"] == v
