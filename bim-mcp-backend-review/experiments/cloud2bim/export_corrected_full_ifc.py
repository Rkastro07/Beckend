from __future__ import annotations
import csv, sys
from pathlib import Path
import ifcopenshell
import ifcopenshell.api as api
import ifcopenshell.util.representation as Representation
from bim_authoring.engine import IfcAuthoringEngine
from corrected_geometry_v2 import build as corrected_geometry, rid
from cloud2bim_patched.space_generator import identify_zones

def create_space(model,body,storey,name,vertices,z,height):
    pts=[model.create_entity("IfcCartesianPoint",Coordinates=(float(x),float(y))) for x,y in vertices]
    if pts[0].Coordinates!=pts[-1].Coordinates:pts.append(pts[0])
    curve=model.create_entity("IfcPolyline",Points=pts)
    profile=model.create_entity("IfcArbitraryClosedProfileDef",ProfileType="AREA",OuterCurve=curve)
    pos=model.create_entity("IfcAxis2Placement3D",Location=model.create_entity("IfcCartesianPoint",Coordinates=(0.,0.,0.)))
    solid=model.create_entity("IfcExtrudedAreaSolid",SweptArea=profile,Position=pos,ExtrudedDirection=model.create_entity("IfcDirection",DirectionRatios=(0.,0.,1.)),Depth=float(height))
    rep=model.create_entity("IfcShapeRepresentation",ContextOfItems=body,RepresentationIdentifier="Body",RepresentationType="SweptSolid",Items=[solid])
    shape=model.create_entity("IfcProductDefinitionShape",Representations=[rep])
    place=model.create_entity("IfcLocalPlacement",RelativePlacement=model.create_entity("IfcAxis2Placement3D",Location=model.create_entity("IfcCartesianPoint",Coordinates=(0.,0.,float(z)))))
    space=api.run("root.create_entity",model,ifc_class="IfcSpace",name=name,predefined_type="INTERNAL")
    space.LongName=f"Room {name}"; space.ObjectPlacement=place; space.Representation=shape
    api.run("aggregate.assign_object",model,products=[space],relating_object=storey)
    return space

def main(src_path,diag_path,out_path):
    model=ifcopenshell.open(str(src_path)); geom,thickness_overrides=corrected_geometry(model)
    thick={}
    with open(diag_path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:thick[r["reference"]]=max(.10,min(.35,float(r["thickness"])))
            except Exception:pass
    before={t:len(model.by_type(t)) for t in ("IfcWall","IfcSlab","IfcDoor","IfcWindow","IfcSpace","IfcStair")}
    # Rejected/obsolete products are removed; slabs and spatial hierarchy remain untouched.
    for typ in ("IfcDoor","IfcWindow","IfcSpace","IfcStair","IfcWall"):
        for product in list(model.by_type(typ)):
            api.run("root.remove_product",model,product=product)
    body=Representation.get_context(model,"Model","Body","MODEL_VIEW")
    if body is None:
        parent=Representation.get_context(model,"Model") or api.run("context.add_context",model,context_type="Model")
        body=api.run("context.add_context",model,context_type="Model",context_identifier="Body",target_view="MODEL_VIEW",parent=parent)
    storeys=model.by_type("IfcBuildingStorey")
    if not storeys:raise RuntimeError("IFC original sem IfcBuildingStorey")
    engine=IfcAuthoringEngine(model); storey=storeys[0]
    for name,(a,b) in sorted(geom.items()):
        source_name=name if name in thick else rid(5)
        wall_thickness=thickness_overrides.get(name,thick.get(source_name,.15))
        engine.create_wall(start=a,end=b,height=3.05,thickness=wall_thickness,body_context=body,storey=storey,elevation=-.159,name=name)
    walls_for_spaces=[]
    for name,(a,b) in geom.items():
        source_name=name if name in thick else rid(5)
        walls_for_spaces.append({"start_point":tuple(a),"end_point":tuple(b),"thickness":thickness_overrides.get(name,thick.get(source_name,.15)),"material":"Wall","z_placement":-.159,"height":3.05,"storey":1})
    zones=identify_zones(walls_for_spaces,snapping_distance=.80,plot_zones=False)
    storey_z=float(storey.Elevation or .141); elevations=sorted(float(s.Elevation) for s in storeys if s.Elevation is not None and float(s.Elevation)>storey_z)
    clear_height=max(.10,(elevations[0]-storey_z-.221) if elevations else 2.967)
    for i,(zone_name,data) in enumerate(sorted(zones.items()),1):
        create_space(model,body,storey,f"1.{i}",data["vertices"],storey_z,clear_height)
    out_path.parent.mkdir(parents=True,exist_ok=True); model.write(str(out_path))
    after={t:len(model.by_type(t)) for t in before}
    print(out_path); print("before",before); print("after",after)

if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
