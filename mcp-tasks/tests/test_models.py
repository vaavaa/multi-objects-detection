from __future__ import annotations

from app.models import ASSIGNEE_NAMES, assignee_display


def test_assignee_names_cover_three_keys():
    assert set(ASSIGNEE_NAMES) == {"serik", "evgeniya", "oksana"}


def test_assignee_display_known():
    assert assignee_display("serik") == "Серик"
    assert assignee_display("evgeniya") == "Евгения"
    assert assignee_display("oksana") == "Оксана"


def test_assignee_display_unknown_passthrough():
    assert assignee_display("unknown") == "unknown"
