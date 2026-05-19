"""Inspeção rápida: áreas, bboxes e Z-range por tipo IFC."""
import sys
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import open3d as o3d


def main():
    ifc_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/ifc/casapequena.ifc"
    print(f"IFC: {ifc_path}")
    m = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)

    for tipo in ["IfcWall", "IfcSlab", "IfcCovering", "IfcRoof", "IfcDoor",
                 "IfcWindow", "IfcColumn", "IfcBeam"]:
        total_area = 0.0
        bboxes = []
        for el in m.by_type(tipo):
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
            except Exception:
                continue
            verts = np.array(shape.geometry.verts, dtype=np.float32).reshape(-1, 3)
            faces = np.array(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
            if len(verts) == 0:
                continue
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            a = mesh.get_surface_area()
            total_area += a
            zmin = float(verts[:, 2].min())
            zmax = float(verts[:, 2].max())
            bboxes.append((zmin, zmax, a))
        if not bboxes:
            continue
        print(f"\n{tipo}: {len(bboxes)} objs, area total {total_area:.1f} m^2")
        for zmin, zmax, a in bboxes[:5]:
            print(f"  Z=[{zmin:.2f}, {zmax:.2f}]  area={a:.1f} m^2")
        if len(bboxes) > 5:
            print(f"  ... +{len(bboxes)-5} mais")


if __name__ == "__main__":
    main()
