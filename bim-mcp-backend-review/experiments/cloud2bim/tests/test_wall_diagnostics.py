import sys
import unittest
from pathlib import Path


PATCHED = Path(__file__).resolve().parents[1] / "cloud2bim_patched"
sys.path.insert(0, str(PATCHED))

from generate_ifc import IFCmodel  # noqa: E402


class WallDiagnosticsIfcTests(unittest.TestCase):
    def test_wall_axis_and_body_use_their_matching_subcontexts(self):
        model = IFCmodel("wall-context-test", "unused.ifc")
        origin = model.ifc_file.create_entity(
            "IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)
        )
        placement = model.ifc_file.create_entity(
            "IfcAxis2Placement3D", Location=origin
        )
        parent_context = model.ifc_file.create_entity(
            "IfcGeometricRepresentationContext",
            ContextIdentifier="Model",
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=0.0001,
            WorldCoordinateSystem=placement,
        )
        model.geom_rep_sub_context = model.ifc_file.create_entity(
            "IfcGeometricRepresentationSubContext",
            ParentContext=parent_context,
            ContextIdentifier="Body",
            ContextType="Model",
            TargetView="MODEL_VIEW",
        )
        model.geom_rep_sub_context_walls = model.ifc_file.create_entity(
            "IfcGeometricRepresentationSubContext",
            ParentContext=parent_context,
            ContextIdentifier="Axis",
            ContextType="Model",
            TargetView="MODEL_VIEW",
        )

        axis = model.wall_axis_representation(
            model.wall_axis_placement((0.0, 0.0), (2.0, 0.0))
        )
        _profile, _solid, body = model.wall_swept_solid_representation(
            (0.0, 0.0),
            (2.0, 0.0),
            2.8,
            0.2,
        )

        self.assertEqual(axis.RepresentationIdentifier, "Axis")
        self.assertEqual(axis.ContextOfItems.ContextIdentifier, "Axis")
        self.assertEqual(body.RepresentationIdentifier, "Body")
        self.assertEqual(body.ContextOfItems.ContextIdentifier, "Body")
        self.assertNotEqual(axis.ContextOfItems.id(), body.ContextOfItems.id())

    def test_wall_reference_and_typed_diagnostics_are_written(self):
        model = IFCmodel("diagnostic-test", "unused.ifc")
        model.owner_history = None
        wall = model.create_wall(
            None,
            None,
            "diagnostic",
            name="W-S01-001",
            tag="W-S01-001",
            object_type="interior",
        )
        properties = [
            model.create_property_single_value("Reference", "W-S01-001"),
            model.create_property_single_value("TopCoverage", 0.58),
            model.create_property_single_value("HeightLayerCount", 6),
            model.create_property_single_value("Accepted", True),
        ]
        property_set, _relation = model.create_property_set(
            wall,
            properties,
            "Pset_Cloud2BIM_WallDiagnostics",
        )

        self.assertEqual(wall.Name, "W-S01-001")
        self.assertEqual(wall.Tag, "W-S01-001")
        self.assertEqual(wall.ObjectType, "interior")
        values = {
            prop.Name: prop.NominalValue.wrappedValue
            for prop in property_set.HasProperties
        }
        self.assertEqual(values["Reference"], "W-S01-001")
        self.assertAlmostEqual(values["TopCoverage"], 0.58)
        self.assertEqual(values["HeightLayerCount"], 6)
        self.assertTrue(values["Accepted"])


if __name__ == "__main__":
    unittest.main()
