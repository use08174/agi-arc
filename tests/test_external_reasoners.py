from arc_agi3.core.types import Action, Frame, GameStatus
from arc_agi3.external import ExternalReasonerHub


def test_external_reasoner_hub_exports_both_reasoners():
    hub = ExternalReasonerHub()
    frame0 = Frame(
        grid=((1, 1), (0, 0)),
        status=GameStatus.IN_PROGRESS,
        info={"guid": "demo-guid"},
    )
    notes0 = hub.reset("demo-task", frame0)
    assert "external_compressarc" in notes0
    assert "external_arcmdl" in notes0
    assert notes0["external_compressarc"]["available"] is True
    assert notes0["external_arcmdl"]["available"] is True
    assert notes0["external_arcmdl"]["mode"] in {"image_only", "compiled_cli", "compiled_unstructured", "compiled_failed"}
    assert "task_json" in notes0["external_arcmdl"]

    frame1 = Frame(
        grid=((1, 0), (0, 1)),
        status=GameStatus.IN_PROGRESS,
        info={"guid": "demo-guid"},
    )
    notes1 = hub.observe_transition(Action("ACTION1"), frame1)
    assert notes1["external_reasoner_metadata"]["observed_frames"] == 2
    assert notes1["external_reasoner_metadata"]["observed_actions"] == 1
    assert any(item.startswith("compressarc:") for item in notes1["external_reasoner_summary"])
    assert any(item.startswith("arcmdl:") for item in notes1["external_reasoner_summary"])
