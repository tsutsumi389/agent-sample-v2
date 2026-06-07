"""テスト共通の fixture。"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def make_item():
    """store の検索結果アイテムを模した最小オブジェクト（.value を持つ）を作るファクトリ。"""
    return lambda value: SimpleNamespace(value=value)
