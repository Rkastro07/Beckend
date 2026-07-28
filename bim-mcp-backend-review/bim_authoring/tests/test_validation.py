import unittest

from bim_authoring.validation import validate_filling


class FakeEntity:
    _next_id = 1

    def __init__(self, ifc_class, **attributes):
        self.ifc_class = ifc_class
        self._id = FakeEntity._next_id
        FakeEntity._next_id += 1
        for name, value in attributes.items():
            setattr(self, name, value)

    def id(self):
        return self._id

    def is_a(self):
        return self.ifc_class


class ValidationTests(unittest.TestCase):
    def valid_assembly(self):
        host = FakeEntity("IfcWall")
        opening_placement = object()
        opening = FakeEntity(
            "IfcOpeningElement",
            Representation=object(),
            ObjectPlacement=opening_placement,
        )
        opening.VoidsElements = [
            FakeEntity("IfcRelVoidsElement", RelatingBuildingElement=host)
        ]
        fill_placement = FakeEntity(
            "IfcLocalPlacement",
            PlacementRelTo=opening_placement,
        )
        window = FakeEntity(
            "IfcWindow",
            Representation=object(),
            ObjectPlacement=fill_placement,
            OverallWidth=1.2,
            OverallHeight=1.1,
            ContainedInStructure=[FakeEntity("IfcRelContainedInSpatialStructure")],
        )
        window.FillsVoids = [
            FakeEntity("IfcRelFillsElement", RelatingOpeningElement=opening)
        ]
        return window

    def test_valid_window_passes(self):
        report = validate_filling(self.valid_assembly(), expected_class="IfcWindow")
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.facts["host_class"], "IfcWall")

    def test_missing_fill_relation_is_an_error(self):
        window = self.valid_assembly()
        window.FillsVoids = []
        report = validate_filling(window)
        self.assertFalse(report.ok)
        self.assertEqual(report.issues[0].code, "FILL_RELATION_COUNT")


if __name__ == "__main__":
    unittest.main()
