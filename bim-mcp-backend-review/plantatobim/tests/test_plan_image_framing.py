import cv2
import numpy as np
from plantatobim.plan_image_framing import detect_building_bbox


def test_crop_removes_page_frame_and_separate_title_block():
    image = np.full((1000, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (4, 4), (795, 995), (0, 0, 0), 1)
    cv2.rectangle(image, (170, 170), (630, 710), (0, 0, 0), 8)
    cv2.line(image, (170, 430), (630, 430), (0, 0, 0), 8)
    cv2.rectangle(image, (40, 890), (760, 970), (0, 0, 0), 1)
    x1, y1, x2, y2 = detect_building_bbox(image)
    assert 0 < x1 < 166 and 0 < y1 < 166
    assert 634 < x2 < 800 and 714 < y2 < 890


def test_crop_keeps_an_attached_wing():
    image = np.full((1000, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (797, 997), (0, 0, 0), 1)
    points = np.array([(180, 150), (500, 150), (500, 600),
                       (700, 650), (700, 800), (180, 800)])
    cv2.polylines(image, [points], True, (0, 0, 0), 5)
    x1, y1, x2, y2 = detect_building_bbox(image)
    assert x1 < 177 and y1 < 147 and x2 > 703 and y2 > 803


def test_blank_image_falls_back_to_full_page():
    assert detect_building_bbox(np.full((200, 300, 3), 255, np.uint8)) == (0, 0, 300, 200)


def test_separate_substantial_views_fall_back_without_losing_a_wing():
    image = np.full((1200, 1000, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (990, 1190), (0, 0, 0), 1)
    cv2.rectangle(image, (100, 100), (550, 600), (0, 0, 0), 6)
    cv2.rectangle(image, (600, 850), (940, 1150), (0, 0, 0), 6)
    assert detect_building_bbox(image) == (0, 0, 1000, 1200)


def test_full_bleed_thick_external_walls_are_not_a_page_frame():
    image = np.full((1000, 800, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (790, 990), (0, 0, 0), 10)
    cv2.rectangle(image, (120, 120), (600, 700), (0, 0, 0), 5)
    x1, y1, x2, y2 = detect_building_bbox(image)
    assert x1 <= 5 and y1 <= 5 and x2 >= 795 and y2 >= 995
