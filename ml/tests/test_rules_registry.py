from vision.rules.base import RULES, Rule, register

def test_register_adds_rule():
    @register("dummy")
    class DummyRule(Rule):
        incident_type = "dummy"

        def update(self, ctx):
            return[]

    assert RULES["dummy"] is DummyRule
    rule = RULES["dummy"]("cam-01", [])
    assert rule.camera_id == "cam-01"
    assert rule.update(None) == []