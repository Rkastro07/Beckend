"""Image framing only: no wall/opening candidates or ML models."""
from __future__ import annotations

import cv2
import numpy as np


def interval_overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def detect_building_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find thick, long ink while ignoring dimensions, text and view arrows."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Thin CAD drawings often have no thick wall core, but their architectural
    # contours enclose large regions. Join overlapping contour envelopes while
    # discarding sparse dimension lines.
    contour_binary = (gray < 180).astype(np.uint8) * 255
    contour_binary = cv2.morphologyEx(
        contour_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    raw_contours = cv2.findContours(
        contour_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    contour_boxes: list[tuple[int, int, int, int, float]] = []
    image_area = float(width * height)
    for contour in raw_contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = float(box_width * box_height)
        fill = float(cv2.contourArea(contour)) / max(1.0, box_area)
        # A closed page frame used to win the area ranking and absorb every
        # room, legend and title block. Ignore only near-page-sized, thin
        # rectangular contours; an irregular plan near the edges is preserved.
        if (box_width > width * 0.85 and box_height > height * 0.85
                and fill > 0.92):
            border = max(1, round(min(width, height) * 0.004))
            core = contour_binary[y + border:y + box_height - border,
                                  x + border:x + box_width - border]
            inset = max(3, round(min(width, height) * 0.003))
            ix1, iy1 = x + inset, y + inset
            ix2, iy2 = x + box_width - inset - 1, y + box_height - inset - 1
            edge_support = np.mean([
                np.mean(contour_binary[iy1, ix1:ix2] > 0),
                np.mean(contour_binary[iy2, ix1:ix2] > 0),
                np.mean(contour_binary[iy1:iy2, ix1] > 0),
                np.mean(contour_binary[iy1:iy2, ix2] > 0),
            ])
            # Dense drawings filling the sheet should not be cropped.
            if core.size and np.count_nonzero(core) / core.size < 0.25 and edge_support < 0.5:
                continue
        touches_page_edge = (x < width * 0.01 or y < height * 0.01
                             or x + box_width > width * 0.99
                             or y + box_height > height * 0.99)
        if touches_page_edge and fill < 0.20:
            continue
        if box_area / image_area >= 0.02 and fill >= 0.045:
            contour_boxes.append((x, y, x + box_width, y + box_height, float(cv2.contourArea(contour))))

    contour_groups: list[list[tuple[int, int, int, int, float]]] = []
    for box in sorted(contour_boxes, key=lambda item: item[4], reverse=True):
        matching_groups = []
        for group in contour_groups:
            if any(
                box[0] <= other[2] + 5
                and box[2] + 5 >= other[0]
                and box[1] <= other[3] + 5
                and box[3] + 5 >= other[1]
                for other in group
            ):
                matching_groups.append(group)
        if not matching_groups:
            contour_groups.append([box])
            continue
        primary = matching_groups[0]
        primary.append(box)
        for extra in matching_groups[1:]:
            primary.extend(extra)
            contour_groups.remove(extra)

    contour_candidate: tuple[int, int, int, int] | None = None
    if contour_groups:
        def group_bounds(
            group: list[tuple[int, int, int, int, float]],
        ) -> tuple[int, int, int, int]:
            return (
                min(item[0] for item in group),
                min(item[1] for item in group),
                max(item[2] for item in group),
                max(item[3] for item in group),
            )

        def box_area(box: tuple[int, int, int, int]) -> float:
            return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))

        def group_score(group: list[tuple[int, int, int, int, float]]) -> float:
            bounds = group_bounds(group)
            return box_area(bounds) + sum(item[4] for item in group) * 0.15

        best_group = max(contour_groups, key=group_score)
        best_bounds = group_bounds(best_group)
        best_area = box_area(best_bounds)

        # A planta pode ter alas separadas por circulacoes abertas. Selecionar
        # apenas o maior componente corta metade do pavimento. Agregamos grupos
        # arquitetonicos relevantes que compartilham a mesma faixa horizontal
        # ou vertical com o grupo principal; blocos de legenda isolados ficam de
        # fora por tamanho e falta de sobreposicao.
        selected_bounds: list[tuple[int, int, int, int]] = []
        for group in contour_groups:
            bounds = group_bounds(group)
            area = box_area(bounds)
            horizontal_overlap = interval_overlap(
                float(bounds[0]), float(bounds[2]),
                float(best_bounds[0]), float(best_bounds[2]),
            ) / max(1.0, min(bounds[2] - bounds[0], best_bounds[2] - best_bounds[0]))
            vertical_overlap = interval_overlap(
                float(bounds[1]), float(bounds[3]),
                float(best_bounds[1]), float(best_bounds[3]),
            ) / max(1.0, min(bounds[3] - bounds[1], best_bounds[3] - best_bounds[1]))
            relevant_size = area >= max(image_area * 0.018, best_area * 0.07)
            aligned_with_plan = max(horizontal_overlap, vertical_overlap) >= 0.20
            gap = max(0, bounds[0] - best_bounds[2], best_bounds[0] - bounds[2],
                      bounds[1] - best_bounds[3], best_bounds[1] - bounds[3])
            nearby = gap <= min(width, height) * 0.10
            aspect = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / max(
                1, min(bounds[2] - bounds[0], bounds[3] - bounds[1]))
            if group is not best_group and not nearby and area >= best_area * 0.25 and aspect < 6:
                # A second substantial view/wing is ambiguous: retain the
                # whole image rather than silently discarding architecture.
                return 0, 0, width, height
            if group is best_group or (relevant_size and aligned_with_plan and nearby):
                selected_bounds.append(bounds)

        if selected_bounds:
            pad = max(8, int(round(min(width, height) * 0.015)))
            contour_candidate = (
                max(0, min(item[0] for item in selected_bounds) - pad),
                max(0, min(item[1] for item in selected_bounds) - pad),
                min(width, max(item[2] for item in selected_bounds) + pad),
                min(height, max(item[3] for item in selected_bounds) + pad),
            )
            if box_area(contour_candidate) / image_area >= 0.15:
                return contour_candidate

    dark = (gray < 135).astype(np.uint8)
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    thick = (distance >= 1.35).astype(np.uint8) * 255
    horizontal_size = max(7, int(round(width * 0.018)))
    vertical_size = max(7, int(round(height * 0.018)))
    horizontal = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1)),
    )
    vertical = cv2.morphologyEx(
        thick,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size)),
    )
    structural = cv2.bitwise_or(horizontal, vertical)
    if np.count_nonzero(structural) < 40:
        return 0, 0, width, height

    # Join wall fragments across normal door/window gaps. The largest resulting
    # component is the building; isolated title text and view arrows stay out.
    join_width = max(25, int(round(min(width, height) * 0.12)))
    join_height = max(17, int(round(min(width, height) * 0.075)))
    if join_width % 2 == 0:
        join_width += 1
    if join_height % 2 == 0:
        join_height += 1
    connected = cv2.morphologyEx(
        structural,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_width, join_height)),
    )
    contours = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return 0, 0, width, height
    building = max(contours, key=cv2.contourArea)
    x, y, box_width, box_height = cv2.boundingRect(building)
    pad = max(1, int(round(min(width, height) * 0.002)))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(width, x + box_width + pad)
    bottom = min(height, y + box_height + pad)
    if (right - left) * (bottom - top) < width * height * 0.12:
        return 0, 0, width, height
    return left, top, right, bottom
