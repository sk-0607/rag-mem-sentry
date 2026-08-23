from pathlib import Path

from app import config
from memory.store import ExperienceStore, default_experience_store


def test_factory_returns_an_experience_store(tmp_path):
    assert isinstance(default_experience_store(tmp_path), ExperienceStore)


def test_factory_places_the_jsonl_mirror_under_the_data_dir(tmp_path):
    store = default_experience_store(tmp_path)
    assert store.jsonl_path == tmp_path / "experiences.jsonl"


def test_factory_uses_the_configured_collection(tmp_path):
    assert default_experience_store(tmp_path).collection_name == config.EXPERIENCE_COLLECTION


def test_jsonl_path_is_a_path_not_a_str(tmp_path):
    # _append_jsonl calls .parent; a bare str from config would AttributeError
    # only at the first write, long after construction.
    assert isinstance(default_experience_store(tmp_path).jsonl_path, Path)


def test_chroma_dir_is_a_path_not_a_str(tmp_path):
    assert isinstance(default_experience_store(tmp_path).chroma_dir, Path)


def test_a_str_data_dir_is_accepted(tmp_path):
    assert default_experience_store(str(tmp_path)).jsonl_path == tmp_path / "experiences.jsonl"
