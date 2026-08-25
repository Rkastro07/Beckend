import csv
import json
import os
import sys

from aux_functions import *
from generate_ifc import IFCmodel
from space_generator import *

# === Load Configuration ===
config = load_config_and_variables()

# === Assign variables ===
e57_input = config["e57_input"]
if e57_input:
    e57_file_names = config["e57_file_names"]
xyz_filenames = config["xyz_filenames"]
exterior_scan = config["exterior_scan"]
dilute_pointcloud = config["dilute_pointcloud"]
dilution_factor = config["dilution_factor"]
pc_resolution = config["pc_resolution"]
grid_coefficient = config["grid_coefficient"]

bfs_thickness = config["bfs_thickness"]
tfs_thickness = config["tfs_thickness"]

min_wall_length = config["min_wall_length"]
min_wall_thickness = config["min_wall_thickness"]
max_wall_thickness = config["max_wall_thickness"]
exterior_walls_thickness = config["exterior_walls_thickness"]

ifc_output_file = config["ifc_output_file"]
ifc_project_name = config["ifc_project_name"]
ifc_project_long_name = config["ifc_project_long_name"]
ifc_project_version = config["ifc_project_version"]

ifc_author_name = config["ifc_author_name"]
ifc_author_surname = config["ifc_author_surname"]
ifc_author_organization = config["ifc_author_organization"]

ifc_building_name = config["ifc_building_name"]
ifc_building_type = config["ifc_building_type"]
ifc_building_phase = config["ifc_building_phase"]

ifc_site_latitude = config["ifc_site_latitude"]
ifc_site_longitude = config["ifc_site_longitude"]
ifc_site_elevation = config["ifc_site_elevation"]
material_for_objects = config["material_for_objects"]

# Tiled Cloud-to-BIM only needs the geometric wall sidecar from each tile.
# In this mode we deliberately skip the legacy opening/space passes and all
# IFC authoring; the final model is authored once, after global stitching.
geometry_only = os.environ.get('CLOUD2BIM_GEOMETRY_ONLY', '0') == '1'


def write_wall_diagnostics(walls, all_openings, output='wall_diagnostics.csv'):
    diagnostic_fields = [
        'reference', 'storey', 'evidence_type', 'detector', 'confidence',
        'review_status', 'length', 'thickness', 'start_x', 'start_y', 'end_x',
        'end_y', 'height_layers', 'height_band_min', 'height_band_max',
        'accepted_face', 'bottom_coverage', 'top_coverage',
        'persistent_coverage', 'second_face_persistent_coverage',
        'paired_persistent_coverage', 'pair_coherence', 'upper_run_coverage',
        'profile_source', 'profile_decision', 'profile_point_count',
        'measured_thickness', 'thickness_mad', 'face_rms',
        'longitudinal_bins', 'longitudinal_bin_size', 'face_band', 'raster_score',
        'detection_score', 'point_count',
        'final_vertical_slices', 'opening_count',
    ] + ['layer_%02d_coverage' % layer for layer in range(1, 13)]
    with open(output, 'w', newline='', encoding='utf-8') as diagnostic_stream:
        diagnostic_writer = csv.DictWriter(
            diagnostic_stream, fieldnames=diagnostic_fields)
        diagnostic_writer.writeheader()
        for wall_record in walls:
            diagnostic = dict(wall_record.get('diagnostics') or {})
            layers = list(diagnostic.pop('layer_coverage', []) or [])
            row = {
                'reference': wall_record.get('reference'),
                'storey': wall_record.get('storey'),
                'evidence_type': wall_record.get('label'),
                'opening_count': sum(
                    opening['opening_wall_id'] == wall_record['wall_id']
                    for opening in all_openings),
                **diagnostic,
            }
            for layer, coverage in enumerate(layers, start=1):
                if layer <= 12:
                    row['layer_%02d_coverage' % layer] = coverage
            diagnostic_writer.writerow({
                field: row.get(field, '') for field in diagnostic_fields})

# Optional contract from the interactive preview. Storeys present in this
# file use exactly the approved axes; absent storeys keep automatic detection.
wall_overrides = {}
wall_override_file = os.environ.get('WALL_OVERRIDE_FILE')
if wall_override_file:
    try:
        with open(wall_override_file, 'r', encoding='utf-8') as override_stream:
            override_payload = json.load(override_stream)
        wall_overrides = {
            int(key): value
            for key, value in override_payload.get('storeys', {}).items()
        }
        print('Preview wall overrides loaded for storeys: %s' %
              sorted(wall_overrides.keys()))
    except Exception as override_error:
        raise RuntimeError('invalid WALL_OVERRIDE_FILE: %s' % override_error)

# === Static Settings ===
# colours for model
door_colour_rgb = (0.541, 0.525, 0.486)
window_colour_rgb = (0.761, 0.933, 1.0)
column_colour_rgb = (0.596,0.576,1.0)
beam_colour_rgb =  (0.157,0.478,0.0)
stair_colour_rgb = (0.992, 0.270, 0.153)
# Spaces are volumetric helpers, not building elements.  Keeping their IFC
# representation translucent prevents them from visually hiding valid walls
# in viewers that render IfcSpace bodies by default.
space_colour_rgb = (0.650, 0.850, 1.0)

# === Logger ===
last_time = time.time()
log_filename = "log.txt"

# SECTION: Import Point Clouds

# read e57 files and create xyz
if e57_input:
    for (idx, e57_file_name) in enumerate(e57_file_names):
        last_time = log('Reading %s.' % e57_file_name, last_time, log_filename)
        imported_e57_data = read_e57(e57_file_name)
        e57_data_to_xyz(imported_e57_data, xyz_filenames[idx], chunk_size=1e10)
        last_time = log('File %s converted to ASCII format, saved as %s.' % (e57_file_name, xyz_filenames[idx]),
                        last_time, log_filename)

# read xyz file
points_xyz, points_rgb = np.empty((0, 3)), np.empty((0, 3))
for xyz_filename in xyz_filenames:
    last_time = log('Extracting data from %s...' % xyz_filename, last_time, log_filename)
    points_xyz_temp, points_rgb_temp = load_xyz_file(xyz_filename, plot_xyz=False, select_ith_lines=dilute_pointcloud,
                                                     ith_lines=dilution_factor)
    points_xyz = np.vstack((points_xyz, np.array(points_xyz_temp)))
    # points_rgb = np.vstack((points_rgb, np.array(points_rgb_temp)))
# Keep the source coordinates intact.  Rounding the complete cloud before
# rasterisation can move points across pixel and height-slice boundaries; a
# valid pair of wall faces may then disappear before the original-point TLS
# refinement has a chance to measure it.
points_xyz = np.asarray(points_xyz, dtype=float)
last_time = log('All point cloud data imported.', last_time, log_filename)

# SECTION: Segment Slabs and Split the Point Cloud to Storeys
print("-" * 50)
print("Slab segmentation")
print("-" * 50)
# scan the model along the z-coordinate and search for planes parallel to xy-plane
slabs, horizontal_surface_planes = identify_slabs(points_xyz, points_rgb, bfs_thickness,
                                                  tfs_thickness, z_step=0.15,
                                                  pc_resolution=pc_resolution,
                                                  plot_segmented_plane=False)  # plot with open 3D

# Quando a geometria foi aprovada no editor, os níveis do histograma completo
# são mais confiáveis que uma nova detecção na amostra comprimida. Sem este
# contrato, uma amostra esparsa pode encontrar apenas o teto, produzir uma laje
# isolada e, consequentemente, zero pavimentos/IfcWall.
for storey_index, override in wall_overrides.items():
    approved_model = override.get('modelo') or {}
    approved_slab = approved_model.get('laje') or {}
    contour = approved_slab.get('contorno') or []
    vertical = override.get('vertical') or {}
    if (len(contour) < 3
            or vertical.get('floor_bottom_z') is None
            or vertical.get('ceiling_bottom_z') is None):
        continue

    xs = [float(value[0]) for value in contour]
    ys = [float(value[1]) for value in contour]
    polygon = Polygon(list(zip(xs, ys)))
    floor_cfg = approved_slab.get('piso') or {}
    ceiling_cfg = approved_slab.get('teto') or {}

    def _approved_slab_record(bottom_z, thickness):
        return {
            'polygon': polygon,
            'polygon_x_coords': list(xs),
            'polygon_y_coords': list(ys),
            'components': [],
            'slab_bottom_z_coord': float(bottom_z),
            'thickness': float(thickness),
            'detector': 'approved-editor-vertical',
        }

    while len(slabs) <= storey_index + 1:
        slabs.append(_approved_slab_record(
            vertical['ceiling_bottom_z'],
            ceiling_cfg.get('espessura', tfs_thickness),
        ))
    slabs[storey_index] = _approved_slab_record(
        vertical['floor_bottom_z'],
        floor_cfg.get('espessura', bfs_thickness),
    )
    slabs[storey_index + 1] = _approved_slab_record(
        vertical['ceiling_bottom_z'],
        ceiling_cfg.get('espessura', tfs_thickness),
    )

# O detector continua propondo os niveis. Quando o cliente aprovou o modelo no
# editor, o contorno/espessura/altura exibidos na previa passam a ser a fonte da
# verdade para o pavimento selecionado.
for storey_index, override in wall_overrides.items():
    if not (0 <= storey_index < len(slabs) - 1):
        continue
    approved_model = override.get('modelo') or {}
    approved_slab = approved_model.get('laje') or {}
    contour = approved_slab.get('contorno') or []
    if len(contour) >= 3:
        xs = [float(value[0]) for value in contour]
        ys = [float(value[1]) for value in contour]
        for slab_index in (storey_index, storey_index + 1):
            slabs[slab_index]['polygon_x_coords'] = list(xs)
            slabs[slab_index]['polygon_y_coords'] = list(ys)
            try:
                slabs[slab_index]['polygon'] = Polygon(list(zip(xs, ys)))
            except Exception:
                pass
    floor_cfg = approved_slab.get('piso') or {}
    ceiling_cfg = approved_slab.get('teto') or {}
    if floor_cfg.get('espessura') is not None:
        slabs[storey_index]['thickness'] = float(floor_cfg['espessura'])
    if ceiling_cfg.get('espessura') is not None:
        slabs[storey_index + 1]['thickness'] = float(ceiling_cfg['espessura'])
    slabs[storey_index]['skip_ifc'] = not bool(floor_cfg.get('ativo', True))
    slabs[storey_index + 1]['skip_ifc'] = not bool(ceiling_cfg.get('ativo', True))
    approved_cfg = override.get('config') or {}
    approved_walls = approved_model.get('paredes') or []
    approved_wall_tops = []
    for approved_wall in approved_walls:
        if approved_wall.get('altura') is None:
            continue
        approved_wall_tops.append(
            float(approved_wall.get('elevacao', 0.0))
            + float(approved_wall['altura']))
    if approved_wall_tops:
        # O teto segue o nivel predominante aprovado no editor. Isso evita que
        # um valor global antigo descole a laje de todas as paredes individuais.
        ceiling_height = float(np.median(approved_wall_tops))
    elif approved_cfg.get('altura') is not None:
        ceiling_height = float(approved_cfg['altura'])
    else:
        ceiling_height = None
    if ceiling_height is not None:
        floor_base = float(slabs[storey_index]['slab_bottom_z_coord'])
        slabs[storey_index + 1]['slab_bottom_z_coord'] = (
            floor_base + ceiling_height)

# SECTION: Segment Walls and Classify Openings
print("-" * 50)
print("Wall segmentation")
print("-" * 50)

# merge_horizontal_pointclouds_in_storey(horizontal_surface_planes)
point_cloud_storeys = split_pointcloud_to_storeys(points_xyz, slabs)
# display_cross_section_plot(point_cloud_storeys, slabs)
walls, all_openings, zones = [], [], []
wall_id = 0
for i, storey_pointcloud in enumerate(point_cloud_storeys):

    if exterior_scan:
        z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
        wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement
    else:
        if i == 0:
            z_placement = slabs[i]['slab_bottom_z_coord']
            if i == len(point_cloud_storeys) - 1:
                wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement + tfs_thickness
            else:
                wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement
        elif i == len(point_cloud_storeys) - 1:
            z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
            wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement + tfs_thickness
        else:
            z_placement = slabs[i]['slab_bottom_z_coord'] + slabs[i]['thickness']
            wall_height = slabs[i + 1]['slab_bottom_z_coord'] - z_placement + slabs[i + 1]['thickness']

    top_z_placement = slabs[i + 1]['slab_bottom_z_coord']

    wall_override = wall_overrides.get(i)
    approved_model = (wall_override or {}).get('modelo') or {}
    approved_wall_metadata = approved_model.get('paredes') or []
    approved_cfg = (wall_override or {}).get('config') or {}
    if wall_override is not None and approved_cfg.get('altura') is not None:
        wall_height = float(approved_cfg['altura'])
    # aberturas aprovadas no editor: {eixo_idx: [ {tipo, s_centro, largura}, ... ]}
    override_openings_by_axis = {}
    # True só quando o cliente passou pelo editor (lista presente, mesmo vazia):
    # aí as esquadrias manuais são a verdade e a detecção automática é desligada.
    override_openings_manual = (
        wall_override is not None and wall_override.get('aberturas') is not None)
    if wall_override is not None:
        override_axes = wall_override.get('eixos', [])
        start_points = [(row[0], row[1]) for row in override_axes]
        end_points = [(row[2], row[3]) for row in override_axes]
        wall_thicknesses = [max(float(row[4]), 0.03) for row in override_axes]
        wall_labels = [str(row[5]) for row in override_axes]
        wall_materials = ['Concrete'] * len(override_axes)
        wall_diagnostics = []
        for axis_index, (start, end, thickness, label) in enumerate(zip(
                start_points, end_points, wall_thicknesses, wall_labels)):
            wall_diagnostics.append({
                'detector': 'preview',
                'evidence_type': label,
                'height_layers': 0,
                'height_band_min': float(z_placement),
                'height_band_max': float(top_z_placement),
                'accepted_face': -1,
                'bottom_coverage': 0.0,
                'top_coverage': 0.0,
                'persistent_coverage': 0.0,
                'detection_score': 1.0,
                'layer_coverage': [],
                'review_status': 'MANUAL_APPROVED',
                'confidence': 'REVIEWED',
                'point_count': 0,
                'final_vertical_slices': 0,
                'length': float(np.hypot(
                    end[0] - start[0], end[1] - start[1])),
                'thickness': float(thickness),
                'start_x': float(start[0]),
                'start_y': float(start[1]),
                'end_x': float(end[0]),
                'end_y': float(end[1]),
            })
        wall_axes = list(zip(start_points, end_points))
        translated_filtered_rotated_wall_groups = prepare_wall_points_from_axes(
            storey_pointcloud, wall_axes, wall_thicknesses,
            z_placement, top_z_placement)
        # config de alturas (mesmos defaults do plantatobim)
        _cfg = approved_cfg
        _porta_h = float(_cfg.get('porta_altura', 2.1))
        _jan_h = float(_cfg.get('janela_altura', 1.2))
        _jan_peit = float(_cfg.get('janela_peitoril', 1.0))
        for _ab in wall_override.get('aberturas', []) or []:
            try:
                _idx = int(_ab['eixo_idx'])
                _s = float(_ab['s_centro'])
                _w = float(_ab['largura'])
                _tp = 'window' if str(_ab.get('tipo')) == 'window' else 'door'
                _individual_h = float(_ab.get(
                    'altura', _jan_h if _tp == 'window' else _porta_h))
                if _tp == 'window':
                    _individual_sill = float(_ab.get('peitoril', _jan_peit))
                    _zmin, _zmax = _individual_sill, _individual_sill + _individual_h
                else:
                    _zmin, _zmax = 0.0, _individual_h
                override_openings_by_axis.setdefault(_idx, []).append(
                    ((_s - _w / 2, _s + _w / 2), (_zmin, _zmax), _tp))
            except (KeyError, ValueError, TypeError):
                continue
        print('Storey %d: using %d wall axes approved in preview (%d aberturas manuais)' %
              (i + 1, len(override_axes), sum(len(v) for v in override_openings_by_axis.values())))
    else:
        (start_points, end_points, wall_thicknesses, wall_materials,
         translated_filtered_rotated_wall_groups, wall_labels,
         wall_diagnostics) = (
            identify_walls(storey_pointcloud, pc_resolution, min_wall_length, min_wall_thickness, max_wall_thickness,
                           z_placement, top_z_placement, grid_coefficient, slabs[i + 1]['polygon'], exterior_scan,
                           exterior_walls_thickness=exterior_walls_thickness))

    print("-" * 50)
    print("Rectangular openings detection")
    print("-" * 50)
    os.makedirs("images", exist_ok=True)
    os.makedirs("images/pdf", exist_ok=True)
    os.makedirs("images/pdf", exist_ok=True)
    os.makedirs("images/wall_outputs_images/", exist_ok=True)
    for j in range(len(start_points)):
        # descarta parede degenerada: pareamento pode gerar NaN/inf ou eixo nulo
        # (quebrava o identify_zones no shapely e sujava o IFC)
        _sp, _ep = start_points[j], end_points[j]
        _vals = [_sp[0], _sp[1], _ep[0], _ep[1], wall_thicknesses[j]]
        if not all(np.isfinite(v) for v in _vals) or \
                np.hypot(_ep[0] - _sp[0], _ep[1] - _sp[1]) < 0.05:
            continue
        wall_id += 1
        wall_reference = 'W-S%02d-%03d' % (i + 1, wall_id)
        wall_diagnostic = (
            dict(wall_diagnostics[j])
            if j < len(wall_diagnostics)
            else {
                'detector': 'unknown',
                'evidence_type': wall_labels[j],
                'confidence': 'UNSCORED',
                'review_status': 'AUTO_ACCEPTED',
                'layer_coverage': [],
            }
        )
        approved_wall = (
            approved_wall_metadata[j]
            if j < len(approved_wall_metadata) else {})
        approved_height = float(approved_wall.get('altura', wall_height))
        # No editor Scan-to-BIM a cota individual e relativa ao piso visual
        # (Z=0). Converte para a cota absoluta da nuvem somente na exportacao.
        approved_elevation = (
            float(z_placement) + float(approved_wall['elevacao'])
            if approved_wall.get('elevacao') is not None
            else float(z_placement))
        walls.append({'wall_id': wall_id, 'storey': i + 1, 'start_point': start_points[j], 'end_point': end_points[j],
                      'thickness': wall_thicknesses[j], 'material': wall_materials[j], 'z_placement': approved_elevation,
                      'height': approved_height, 'label': wall_labels[j],
                      'preview': wall_override is not None,
                      'reference': wall_reference,
                      'diagnostics': wall_diagnostic})

        if j in override_openings_by_axis:
            # o cliente definiu as esquadrias desta parede no editor: usa
            # exatamente essas, sem redetectar (contrato tela->IFC)
            opening_widths = [op[0] for op in override_openings_by_axis[j]]
            opening_heights = [op[1] for op in override_openings_by_axis[j]]
            opening_types = [op[2] for op in override_openings_by_axis[j]]
        elif override_openings_manual or geometry_only:
            # editor: parede aprovada mas sem esquadria marcada = sem abertura
            # (o cliente é a fonte da verdade das portas/janelas)
            opening_widths, opening_heights, opening_types = [], [], []
        else:
            (opening_widths, opening_heights,
             opening_types) = identify_openings(j + 1, translated_filtered_rotated_wall_groups[j],
                                                wall_labels[j], pc_resolution, grid_coefficient,
                                                min_opening_width=0.4, min_opening_height=0.6,
                                                max_opening_aspect_ratio=4, door_z_max=0.1,
                                                door_min_height=1.6, opening_min_z_top=1.6,
                                                plot_histograms_for_openings=False)

        # Temporary list to store openings for the current wall
        wall_openings = []

        # Iterate through the detected openings and store the information
        for (x_start, x_end), (z_min, z_max), opening_type in zip(opening_widths, opening_heights, opening_types):
            opening_info = {
                "opening_wall_id": wall_id,
                "opening_type": opening_type,
                "x_range_start": x_start,
                "x_range_end": x_end,
                "z_range_min": z_min,
                "z_range_max": z_max
            }
            # Append the current opening's information to the wall's openings list
            wall_openings.append(opening_info)

        # After processing all openings for the current wall, append them to the all_openings list
        all_openings.extend(wall_openings)

        # Print or further process the results
        print(f"Wall {j + 1}:")
        for (x_start, x_end), (z_min, z_max), opening_type in zip(opening_widths, opening_heights, opening_types):
            print(
                f"Opening ({opening_type:s}): X-Range: {x_start:.2f} to {x_end:.2f}, Z-Range: {z_min:.2f} to {z_max:.2f}")
        print("-" * 50)

    # SECTION: Split the Storeys to Zones (Spaces in the IFC)
    print('Segmenting the storey to zones (spaces)...')
    print("-" * 50)
    if geometry_only:
        zones_in_storey = {}
        print('[geometry-only] Space detection skipped.')
    else:
        try:
            # A Space may repair only a small corner/junction error; it must not
            # bridge a missing wall.  The former 80 cm tolerance fabricated room
            # closures that were not present in the detected wall graph.
            space_max_snap = float(os.environ.get('SPACE_MAX_SNAP', '0.15'))
            approved_spaces = approved_model.get('spaces') or []
            if wall_override is not None and approved_spaces:
                zones_in_storey = {
                    str(space.get('id') or 'SPACE-%03d' % (index + 1)): {
                        'vertices': [tuple(value) for value in space.get('contorno', [])],
                        'height': float(approved_cfg.get('altura', wall_height)),
                        'area': float(space.get('area', 0.0)),
                        'source': 'approved-editor-snapshot',
                    }
                    for index, space in enumerate(approved_spaces)
                    if len(space.get('contorno', [])) >= 3
                }
            elif wall_override is not None:
                # O editor pode enviar apenas as paredes aprovadas. Spaces sao
                # derivados topologicos: rode o mesmo reconhecedor de ciclos sobre
                # esses eixos em vez de fabricar um Space pelo contorno da laje.
                zones_in_storey = identify_zones(
                    walls, snapping_distance=space_max_snap, plot_zones=False)
                print(
                    '[space] reconhecimento automatico sobre paredes aprovadas: '
                    f'{len(zones_in_storey)} comodo(s) fechado(s).'
                )
                if (bool((approved_cfg.get('forro') or {}).get('ativo', False))
                        and not zones_in_storey):
                    print(
                        '[!] Forro solicitado, mas nenhuma area fechada foi '
                        'reconhecida; nenhum Space geral sera inventado.'
                    )
            else:
                zones_in_storey = identify_zones(
                    walls, snapping_distance=space_max_snap, plot_zones=False)
        except Exception as _ze:
            # zonas sao acessorias: falha aqui nao pode derrubar o pipeline inteiro
            print(f'[!] identify_zones falhou neste pavimento ({_ze}); seguindo sem spaces.')
            zones_in_storey = {}
    zones.append(zones_in_storey)

write_wall_diagnostics(walls, all_openings)
if geometry_only:
    last_time = log(
        '\nGeometry-only wall diagnostics saved to wall_diagnostics.csv.',
        last_time,
        log_filename,
    )
    sys.exit(0)

# SECTION: Generate IFC
print("-" * 50)
print("Generating IFC model")
print("-" * 50)
ifc_model = IFCmodel(ifc_project_name, ifc_output_file)
ifc_model.define_author_information(ifc_author_name + ' ' + ifc_author_surname, ifc_author_organization)
ifc_model.define_project_data(ifc_building_name, ifc_building_type, ifc_building_phase,
                              ifc_project_long_name, ifc_project_version, ifc_author_organization,
                              ifc_author_name, ifc_author_surname, ifc_site_latitude, ifc_site_longitude,
                              ifc_site_elevation)

# Add building storeys and zones
storeys_ifc, slabs_ifc = [], []
space_material, space_material_def_rep = ifc_model.create_material_with_color(
    'Space volume', space_colour_rgb, transparency=0.85)
for idx, slab in enumerate(slabs):
    # define a storey
    slab_position = slab['slab_bottom_z_coord'] + slab['thickness']
    storeys_ifc.append(ifc_model.create_building_storey('Floor %.1f m' % slab_position, slab_position))

    # define a slab
    # Convert separate x and y coordinate lists into a list of coordinate pairs
    points = [[float(x), float(y)] for x, y in zip(slab['polygon_x_coords'], slab['polygon_y_coords'])]

    # Optionally remove duplicate points to avoid redundancy in the polygon
    # This example uses a simple method by converting each pair into a tuple and then back into a list.
    points_no_duplicates = list(dict.fromkeys(tuple(pt) for pt in points))
    points_no_duplicates = [list(pt) for pt in points_no_duplicates]

    # The create_slab function internally creates the slab placement, extrusion, and shape representation.
    if not slab.get('skip_ifc', False):
        slab_entity = ifc_model.create_slab(
            slab_name='Slab %d' % (idx + 1),
            points=points_no_duplicates,
            slab_z_position=round(slab['slab_bottom_z_coord'], 3),
            slab_height=round(slab['thickness'], 3),
            material_name=material_for_objects,
            components=slab.get('components')
        )

        ifc_model.assign_product_to_storey(slab_entity, storeys_ifc[-1])

    # IfcSpace initialization
    if idx < len(zones) and zones[idx]:  # this means there are some zones inside
        ifc_space_placement = ifc_model.space_placement(slab_position)
        if idx != len(slabs) - 1:  # avoid creating zones on the uppermost slab
            # A room starts on top of its floor slab and ends at the underside
            # of the slab above.  ``space_data["height"]`` is the wall height;
            # using it here used to shift the Space top upward by the floor
            # thickness (30 cm in the Kladno scan), masking walls in 3D views.
            space_clear_height = max(
                0.0,
                float(slabs[idx + 1]['slab_bottom_z_coord']) - float(slab_position)
            )
            zone_number = 1
            for space_name, space_data in zones[idx].items():
                ifc_space = ifc_model.create_space(
                    space_data,
                    ifc_space_placement,
                    (idx + 1),
                    zone_number,
                    storeys_ifc[-1],
                    space_clear_height,
                    approved_name=(
                        space_name
                        if str(space_data.get('source', '')).startswith('approved-editor-')
                        else None
                    ),
                )
                ifc_model.assign_material(ifc_space, space_material)
                storey_cfg = (wall_overrides.get(idx) or {}).get('config') or {}
                forro_cfg = storey_cfg.get('forro') or {}
                if bool(forro_cfg.get('ativo', False)):
                    forro_thickness = max(0.01, float(forro_cfg.get('espessura', 0.03)))
                    # A configuracao pode vir de outro pavimento. Nunca deixe
                    # o forro atravessar ou flutuar acima da laje superior.
                    available_height = float(space_clear_height)
                    if available_height > forro_thickness + 0.01:
                        max_forro_height = available_height - forro_thickness
                        requested_height = float(forro_cfg.get(
                            'altura', max_forro_height))
                        forro_height = min(
                            max(0.1, requested_height),
                            max_forro_height,
                        )
                        ifc_model.create_covering(
                            space_data,
                            ifc_space_placement,
                            'Forro-%s' % space_name,
                            ifc_space,
                            forro_height,
                            forro_thickness,
                        )
                zone_number += 1
    else:
        continue

'''# Column definition for IFC
columns_example = [
    {
        "name": "round", # other classes "rect", "steel"
        "storey": 1,
        "start_point": (0.0, 0.0),  # Only X, Y coordinates
        "direction": (0.2, 0.5),  # Direction only in X, Y plane
        "profile_points": [0.3],  # Square profile [-0.1, -0.1], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]
        "height": 3.0
    }
]

column_material, column_material_def_rep= ifc_model.create_material_with_color("Column material",
                                                                               column_colour_rgb, transparency=0)
column_id=1
for column in columns_example:
    ifc_column = ifc_model.create_column(f"C{column_id:02d}", column['name'], storeys_ifc[column['storey'] - 1], column['start_point'],
                                         column['direction'], column['profile_points'], column['height'])
    ifc_model.assign_material(ifc_column, column_material)
    column_id +=1

# Beams definition for IFC
# Example input parameters
beams_example = [
    {
        "name": "rect",      # A rectangular beam with larger dimensions
        "storey": 2,               # Placed on the second storey
        "start_point": (10.0, 5.0),  # X, Y placement
        "direction": (0.0, -1.0),    # Beam axis direction in XY plane (pointing in negative Y)
        "profile_points": [0.5, 0.7],# Width and height for 'rect'
        "length": 8.0              # Extrusion length along the proper axis (e.g., Z-axis after correction)
    },
    {
        "name": "steel",    # A steel beam with a custom I-shaped profile
        "storey": 2,               # Placed on the second storey
        "start_point": (12.0, 6.0),  # X, Y placement
        "direction": (0.5, 0.5),     # Beam axis direction in XY plane
        "profile_points": [[-0.2, -0.225], [0.2, -0.225], [0.2, -0.165], [0.05, -0.165],
                           [0.05, 0.125], [0.2, 0.125], [0.2, 0.225], [-0.2, 0.225],
                           [-0.2, 0.125], [-0.05, 0.125], [-0.05, -0.165], [-0.2, -0.165],
                           [-0.2, -0.225]],
        "length": 10.0             # Extrusion length
    }
]
beam_material, beam_material_def_rep= ifc_model.create_material_with_color("beam material",
                                                                           beam_colour_rgb)
beam_id=1
for beam in beams_example:
    ifc_model.create_beam(f"B{beam_id:02d}",beam["name"],storeys_ifc[beam["storey"] - 1],beam["start_point"],
                          beam["direction"],beam["profile_points"],beam["length"],beam_material)
    beam_id +=1'''

'''# Stairs definition for IFC
stairs = [
    [  # Curved stair
        {
            "key": "flight_curved",
            "origin": (0.0, 0.0, 0.0),
            "num_risers": 12,
            "raiser_height": 0.17,
            "angle_per_step_deg": 15,
            "inner_radius": 1.0,
            "flight_width": 1.2,
            "storey": 1
        }
    ]
]

stair_material, stair_material_def_rep= ifc_model.create_material_with_color("Stair material",
                                                                               stair_colour_rgb, transparency=0)

for i, stair_parts in enumerate(stairs):
    stair_name = f"Stair_{i+1:03}"
    stair = ifc_model.create_stair(stair_name, storeys_ifc[stair_parts[0]["storey"] - 1], stair_parts, stair_material)
'''
# Wall definition for IFC
for wall in walls:
    wall_record = wall
    start_point = tuple(float(num) for num in wall['start_point'])
    end_point = tuple(float(num) for num in wall['end_point'])
    if start_point == end_point:
        continue
    wall_thickness = wall['thickness']
    wall_material = wall['material']
    wall_z_placement = wall['z_placement']
    wall_heights = wall['height']
    wall_label = wall['label']
    wall_reference = wall['reference']
    wall_diagnostic = dict(wall.get('diagnostics') or {})

    wall_openings = [opening for opening in all_openings if opening['opening_wall_id'] == wall['wall_id']]

    # Create a material layer
    material_layer = ifc_model.create_material_layer(wall_thickness, wall_material)
    # Create an IfcMaterialLayerSet using the material layer (in a list)
    material_layer_set = ifc_model.create_material_layer_set([material_layer])
    # Create an IfcMaterialLayerSetUsage and associate it with the element or product
    material_layer_set_usage = ifc_model.create_material_layer_set_usage(material_layer_set, wall_thickness)
    # Local placement
    wall_placement = ifc_model.wall_placement(wall['z_placement'])
    wall_axis_placement = ifc_model.wall_axis_placement(start_point, end_point)
    wall_axis_representation = ifc_model.wall_axis_representation(wall_axis_placement)
    wall_swept_solid_representation = ifc_model.wall_swept_solid_representation(start_point, end_point, wall_heights,
                                                                                wall_thickness)
    product_definition_shape = ifc_model.product_definition_shape(wall_axis_representation,
                                                                  wall_swept_solid_representation)
    current_story = wall['storey']
    # paredes aprovadas no preview sao marcadas: passos posteriores (aparo de
    # telhado, remocao de fantasma) e a auditoria do job as reconhecem
    wall_description = (
        'preview-locked; reference=%s' % wall_reference
        if wall.get('preview')
        else 'Cloud2BIM V2 detected; reference=%s; confidence=%s' % (
            wall_reference, wall_diagnostic.get('confidence', 'UNSCORED'))
    )
    wall = ifc_model.create_wall(
        wall_placement,
        product_definition_shape,
        wall_description,
        name=wall_reference,
        tag=wall_reference,
        object_type=wall_label,
    )
    assign_material = ifc_model.assign_material(wall, material_layer_set_usage)
    wall_type = ifc_model.create_wall_type(wall, wall_thickness)
    assign_material_2 = ifc_model.assign_material(wall_type[0], material_layer_set)
    assign_object = ifc_model.assign_product_to_storey(wall, storeys_ifc[current_story - 1])
    wall_ext_int_parameter = ifc_model.create_property_single_value(
        "IsExternal", wall_label == 'exterior')
    ifc_model.create_property_set(
        wall, wall_ext_int_parameter, 'wall properties')

    diagnostic_properties = [
        ifc_model.create_property_single_value(
            'Reference', wall_reference),
        ifc_model.create_property_single_value(
            'Detector', wall_diagnostic.get('detector', 'unknown')),
        ifc_model.create_property_single_value(
            'EvidenceType', wall_label),
        ifc_model.create_property_single_value(
            'Confidence', wall_diagnostic.get('confidence', 'UNSCORED')),
        ifc_model.create_property_single_value(
            'ReviewStatus',
            wall_diagnostic.get('review_status', 'AUTO_ACCEPTED')),
        ifc_model.create_property_single_value(
            'HeightLayerCount',
            int(wall_diagnostic.get('height_layers', 0))),
        ifc_model.create_property_single_value(
            'AcceptedFace',
            int(wall_diagnostic.get('accepted_face', -1))),
        ifc_model.create_property_single_value(
            'BottomCoverage',
            float(wall_diagnostic.get('bottom_coverage', 0.0))),
        ifc_model.create_property_single_value(
            'TopCoverage',
            float(wall_diagnostic.get('top_coverage', 0.0))),
        ifc_model.create_property_single_value(
            'PersistentCoverage',
            float(wall_diagnostic.get('persistent_coverage', 0.0))),
        ifc_model.create_property_single_value(
            'SecondFacePersistentCoverage',
            float(wall_diagnostic.get(
                'second_face_persistent_coverage', 0.0))),
        ifc_model.create_property_single_value(
            'PairedPersistentCoverage',
            float(wall_diagnostic.get(
                'paired_persistent_coverage', 0.0))),
        ifc_model.create_property_single_value(
            'PairCoherence',
            float(wall_diagnostic.get('pair_coherence', 0.0))),
        ifc_model.create_property_single_value(
            'UpperRunCoverage',
            float(wall_diagnostic.get('upper_run_coverage', 0.0))),
        ifc_model.create_property_single_value(
            'ProfileSource',
            wall_diagnostic.get('profile_source', 'none')),
        ifc_model.create_property_single_value(
            'ProfileDecision',
            wall_diagnostic.get('profile_decision', 'NOT_SCORED')),
        ifc_model.create_property_single_value(
            'ProfilePointCount',
            int(wall_diagnostic.get('profile_point_count', 0))),
        ifc_model.create_property_single_value(
            'MeasuredThickness',
            float(wall_diagnostic.get('measured_thickness', 0.0))),
        ifc_model.create_property_single_value(
            'ThicknessMAD',
            float(wall_diagnostic.get('thickness_mad', 0.0))),
        ifc_model.create_property_single_value(
            'FaceRMS',
            float(wall_diagnostic.get('face_rms', 0.0))),
        ifc_model.create_property_single_value(
            'LongitudinalBins',
            int(wall_diagnostic.get('longitudinal_bins', 0))),
        ifc_model.create_property_single_value(
            'LongitudinalBinSize',
            float(wall_diagnostic.get('longitudinal_bin_size', 0.0))),
        ifc_model.create_property_single_value(
            'FaceBand',
            float(wall_diagnostic.get('face_band', 0.0))),
        ifc_model.create_property_single_value(
            'RasterScore',
            float(wall_diagnostic.get('raster_score', 0.0))),
        ifc_model.create_property_single_value(
            'DetectionScore',
            float(wall_diagnostic.get('detection_score', 0.0))),
        ifc_model.create_property_single_value(
            'AssignedPointCount',
            int(wall_diagnostic.get('point_count', 0))),
        ifc_model.create_property_single_value(
            'FinalVerticalSlices',
            int(wall_diagnostic.get('final_vertical_slices', 0))),
    ]
    for layer, coverage in enumerate(
            wall_diagnostic.get('layer_coverage', []) or [], start=1):
        diagnostic_properties.append(
            ifc_model.create_property_single_value(
                'Layer%02dCoverage' % layer, float(coverage)))
    ifc_model.create_property_set(
        wall,
        diagnostic_properties,
        'Pset_Cloud2BIM_WallDiagnostics',
    )

    # Create materials
    window_material, window_material_def_rep = ifc_model.create_material_with_color(
        'Window material',
        window_colour_rgb,
        transparency=0.7
    )

    door_material, door_material_def_rep = ifc_model.create_material_with_color(
        'Door material',
        door_colour_rgb
    )

    # Initialize ID counters
    window_id = 1
    door_id = 1

    for opening in wall_openings:
        # Each 'opening' is a dictionary with the opening data
        opening_type = opening['opening_type']
        x_range_start = opening['x_range_start']
        x_range_end = opening['x_range_end']
        z_range_min = opening['z_range_min']
        z_range_max = opening['z_range_max']

        # Assign unique ID based on opening type
        if opening_type == "window":
            opening_id = f"W{window_id:02d}"  # Format as W01, W02, ...
            window_id += 1
        elif opening_type == "door":
            opening_id = f"D{door_id:02d}"  # Format as D01, D02, ...
            door_id += 1
        else:
            print(f"Warning: Unknown opening type: {opening_type}, skipping this opening")
            continue

        # Store the ID in the opening dictionary
        opening['wall_id'] = opening_id

        opening_width = x_range_end - x_range_start
        opening_height = z_range_max - z_range_min
        window_sill_height = z_range_min
        offset_from_start = x_range_start

        opening_closed_profile = ifc_model.opening_closed_profile_def(float(opening_width), wall_thickness)
        opening_placement = ifc_model.opening_placement(start_point, wall_placement)
        opening_extrusion = ifc_model.opening_extrusion(opening_closed_profile, float(opening_height), start_point,
                                                        end_point, float(window_sill_height), float(offset_from_start))
        opening_representation = ifc_model.opening_representation(opening_extrusion)
        opening_product_definition = ifc_model.product_definition_shape_opening(opening_representation)
        wall_opening = ifc_model.create_wall_opening(opening_placement[1], opening_product_definition)
        rel_voids_element = ifc_model.create_rel_voids_element(wall, wall_opening)
        if opening_type == "window":
            window_closed_profile = ifc_model.opening_closed_profile_def(float(opening_width), 0.01)
            window_extrusion = ifc_model.opening_extrusion(window_closed_profile, float(opening_height), start_point,
                                                           end_point, float(window_sill_height), float(offset_from_start))
            window_representation = ifc_model.opening_representation(window_extrusion)
            window_product_definition = ifc_model.product_definition_shape_opening(window_representation)
            window = ifc_model.create_window(
                opening_placement[1], window_product_definition, opening_id,
                overall_height=opening_height, overall_width=opening_width)
            window_type = ifc_model.create_window_type()
            ifc_model.create_rel_defines_by_type(window, window_type)
            ifc_model.create_rel_fills_element(wall_opening, window)
            ifc_model.assign_product_to_storey(window, storeys_ifc[current_story - 1])
            ifc_model.assign_material(window, window_material)
        elif opening_type == "door":
            door_closed_profile = ifc_model.opening_closed_profile_def(float(opening_width), 0.01)
            door_extrusion = ifc_model.opening_extrusion(door_closed_profile, float(opening_height), start_point,
                                                         end_point, float(window_sill_height), float(offset_from_start))
            door_representation = ifc_model.opening_representation(door_extrusion)
            door_product_definition = ifc_model.product_definition_shape_opening(door_representation)
            door = ifc_model.create_door(
                opening_placement[1], door_product_definition, opening_id,
                overall_height=opening_height, overall_width=opening_width)
            ifc_model.create_rel_fills_element(wall_opening, door)
            ifc_model.assign_product_to_storey(door, storeys_ifc[current_story - 1])
            ifc_model.assign_material(door, door_material)

# Write the IFC model to a file
ifc_model.write()
last_time = log('\nIFC model saved to %s.' % ifc_output_file, last_time, log_filename)
