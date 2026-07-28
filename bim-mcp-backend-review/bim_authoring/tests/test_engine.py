import unittest

from bim_authoring.engine import AuthoringError, IfcAuthoringEngine


class FakeEntity:
    _next_id = 1

    def __init__(self, ifc_class):
        self.ifc_class = ifc_class
        self._id = FakeEntity._next_id
        FakeEntity._next_id += 1

    def id(self):
        return self._id

    def is_a(self):
        return self.ifc_class


class FakeApiRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, usecase, model, **kwargs):
        self.calls.append((usecase, kwargs))
        if usecase == "root.create_entity":
            return FakeEntity(kwargs["ifc_class"])
        if usecase.startswith("geometry.add_"):
            return {"representation": usecase}
        return {"relation_or_result": usecase}


class AuthoringEngineTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeApiRunner()
        self.engine = IfcAuthoringEngine(object(), api_runner=self.api)
        self.wall = self.engine.create_wall(
            start=(0, 0),
            end=(5, 0),
            height=2.8,
            thickness=0.15,
            body_context=object(),
            storey=object(),
        )

    def test_window_uses_opening_and_fill_relationships(self):
        result = self.engine.insert_window(
            self.wall,
            offset_from_start=1,
            width=1.2,
            height=1.1,
            sill_height=0.9,
        )
        usecases = [name for name, _ in self.api.calls]
        self.assertIn("geometry.add_window_representation", usecases)
        self.assertIn("feature.add_feature", usecases)
        self.assertIn("feature.add_filling", usecases)
        self.assertLess(
            usecases.index("feature.add_filling"),
            max(i for i, name in enumerate(usecases) if name == "geometry.edit_object_placement"),
        )
        self.assertEqual(result.recipe_id, "assembly.window-in-wall")
        self.assertEqual(result.opening.is_a(), "IfcOpeningElement")
        self.assertEqual(result.filling.is_a(), "IfcWindow")
        self.assertEqual(result.filling.OverallWidth, 1.2)

    def test_door_uses_native_door_representation(self):
        result = self.engine.insert_door(
            self.wall,
            offset_from_start=2,
            width=0.9,
            height=2.1,
        )
        usecases = [name for name, _ in self.api.calls]
        self.assertIn("geometry.add_door_representation", usecases)
        self.assertEqual(result.filling.is_a(), "IfcDoor")

    def test_rejects_opening_outside_wall_before_creating_it(self):
        count_before = len(self.api.calls)
        with self.assertRaisesRegex(AuthoringError, "fim da parede"):
            self.engine.insert_window(
                self.wall,
                offset_from_start=4.5,
                width=1,
                height=1,
                sill_height=1,
            )
        self.assertEqual(len(self.api.calls), count_before)


if __name__ == "__main__":
    unittest.main()
