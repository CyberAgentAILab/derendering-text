import numpy as np
import pyclipper
from shapely.geometry import Polygon


def nms_with_char_cls(
        boxes,
        char_scores,
        overlapThresh,
        neighbourThresh=0.5,
        minScore=0,
        num_neig=0):
    new_boxes = np.zeros_like(boxes)
    new_char_scores = np.zeros_like(char_scores)
    pick = []
    suppressed = [False for _ in range(boxes.shape[0])]
    areas = [
        Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).area
        for b in boxes
    ]
    polygons = pyclipper.scale_to_clipper(boxes[:, :8].reshape((-1, 4, 2)))
    order = boxes[:, 8].argsort()[::-1]
    # print(len(order))
    for _i, i in enumerate(order):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i + 1:]:
                if suppressed[j] is False:
                    try:
                        pc = pyclipper.Pyclipper()
                        pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                        pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                        if len(solution) == 0:
                            inter = 0
                        else:
                            inter = pyclipper.scale_from_clipper(
                                pyclipper.scale_from_clipper(
                                    pyclipper.Area(solution[0])
                                )
                            )
                    except BaseException:
                        inter = 0
                    union = areas[i] + areas[j] - inter
                    iou = inter / union if union > 0 else 0
                    if union > 0 and iou > overlapThresh:
                        suppressed[j] = True
                    if iou > neighbourThresh:
                        neighbours.append(j)
            # print(len(neighbours))
            if len(neighbours) >= num_neig:
                neighbours.append(i)
                temp_scores = (boxes[neighbours, 8] -
                               minScore).reshape((-1, 1))
                new_boxes[i, :8] = (boxes[neighbours, :8] * temp_scores).sum(
                    axis=0
                ) / temp_scores.sum()
                new_boxes[i, 8] = boxes[i, 8]
                new_char_scores[i, :] = (char_scores[neighbours, :] * temp_scores).sum(
                    axis=0
                ) / temp_scores.sum()
            else:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()
    return pick, new_boxes, new_char_scores


def nms(boxes, overlapThresh, neighbourThresh=0.5, minScore=0, num_neig=0):
    new_boxes = np.zeros_like(boxes)
    pick = []
    suppressed = [False for _ in range(boxes.shape[0])]
    areas = [
        Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).area
        for b in boxes
    ]
    polygons = pyclipper.scale_to_clipper(boxes[:, :8].reshape((-1, 4, 2)))
    order = boxes[:, 8].argsort()[::-1]
    for _i, i in enumerate(order):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i + 1:]:
                if suppressed[j] is False:
                    try:
                        pc = pyclipper.Pyclipper()
                        pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                        pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                        if len(solution) == 0:
                            inter = 0
                        else:
                            inter = pyclipper.scale_from_clipper(
                                pyclipper.scale_from_clipper(
                                    pyclipper.Area(solution[0])
                                )
                            )
                    except BaseException:
                        inter = 0
                    union = areas[i] + areas[j] - inter
                    iou = inter / union if union > 0 else 0
                    if union > 0 and iou > overlapThresh:
                        suppressed[j] = True
                    if iou > neighbourThresh:
                        neighbours.append(j)
            if len(neighbours) >= num_neig:
                neighbours.append(i)
                temp_scores = (boxes[neighbours, 8] -
                               minScore).reshape((-1, 1))
                new_boxes[i, :8] = (boxes[neighbours, :8] * temp_scores).sum(
                    axis=0
                ) / temp_scores.sum()
                new_boxes[i, 8] = boxes[i, 8]
            else:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()
    return pick, new_boxes
