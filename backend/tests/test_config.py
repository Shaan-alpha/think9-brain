import pytest

from think9.config import get_settings


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/think9")
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-abc")
    # conftest loads backend/.env, which sets a fitted COVERAGE_TAU. This case is about
    # the built-in default, so the variable has to be absent for it to mean anything.
    monkeypatch.delenv("COVERAGE_TAU", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.database_url == "postgresql://localhost/think9"
    assert settings.drive_folder_id == "folder-abc"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.coverage_tau == 0.5


def test_missing_database_url_fails_loudly(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        get_settings()


def test_coverage_tau_is_overridable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/think9")
    monkeypatch.setenv("COVERAGE_TAU", "0.62")
    get_settings.cache_clear()

    assert get_settings().coverage_tau == 0.62
