# -*- coding: utf-8 -*-
"""Insere o pilar detectado (detect_pilar_v2) como IfcColumn no IFC real."""
import ifcopenshell

FN = 'output_IFC/real_cloud2bim.ifc'
# pilar validado: centro (4.07, 1.25), secao 0.72 x 0.39, pavimento -1.011..1.302
CX, CY = 4.07, 1.25
W, H = 0.72, 0.39
Z_BOT, Z_TOP = -1.011, 1.302

f = ifcopenshell.open(FN)
owner = f.by_type('IfcOwnerHistory')[0]
storey = f.by_type('IfcBuildingStorey')[0]
ctx = f.by_type('IfcGeometricRepresentationContext')[0]

profile = f.create_entity('IfcRectangleProfileDef', ProfileType='AREA',
                          ProfileName='Pilar 72x39', XDim=W, YDim=H)
origin = f.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0))
axis3d = f.create_entity('IfcAxis2Placement3D', Location=origin)
zdir = f.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))
solid = f.create_entity('IfcExtrudedAreaSolid', SweptArea=profile,
                        Position=axis3d, ExtrudedDirection=zdir,
                        Depth=round(Z_TOP - Z_BOT, 3))
shape = f.create_entity('IfcShapeRepresentation', ContextOfItems=ctx,
                        RepresentationIdentifier='Body',
                        RepresentationType='SweptSolid', Items=[solid])
pds = f.create_entity('IfcProductDefinitionShape', Representations=[shape])

loc = f.create_entity('IfcCartesianPoint', Coordinates=(CX, CY, Z_BOT))
place3d = f.create_entity('IfcAxis2Placement3D', Location=loc)
lplace = f.create_entity('IfcLocalPlacement', RelativePlacement=place3d)

col = f.create_entity('IfcColumn', GlobalId=ifcopenshell.guid.new(),
                      OwnerHistory=owner, Name='C01',
                      Description='Pilar detectado por perfil vertical',
                      ObjectPlacement=lplace, Representation=pds)

# pendura no pavimento (reusa a relacao existente)
rel = f.by_type('IfcRelContainedInSpatialStructure')
if rel:
    r = rel[0]
    r.RelatedElements = list(r.RelatedElements) + [col]
else:
    f.create_entity('IfcRelContainedInSpatialStructure',
                    GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner,
                    RelatedElements=[col], RelatingStructure=storey)

f.write(FN)
m = ifcopenshell.open(FN)
print('OK. Conteudo final:')
for t in ['IfcSlab', 'IfcWall', 'IfcWindow', 'IfcDoor', 'IfcColumn']:
    n = len(m.by_type(t))
    if n:
        print(f'  {t:12s} {n}')
