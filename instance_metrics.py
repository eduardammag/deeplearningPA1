import numpy as np


def get_instance_ids(instance_mask):
    """
    Retorna os IDs das instâncias presentes na máscara.

    0 representa background e é ignorado.
    """
    ids = np.unique(instance_mask)

    return ids[ids != 0]


def instance_iou(pred_mask, gt_mask):
    """
    Calcula o IoU entre duas máscaras binárias.
    """

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(
        pred_mask,
        gt_mask
    ).sum()

    union = np.logical_or(
        pred_mask,
        gt_mask
    ).sum()

    if union == 0:
        return 0.0

    return intersection / union


def match_instances(
    pred_instance_mask,
    gt_instance_mask,
    iou_threshold=0.5
):
    """
    Faz matching 1-to-1 entre instâncias previstas e GT.

    O matching é greedy:
    os pares com maior IoU são considerados primeiro.
    """

    pred_ids = get_instance_ids(
        pred_instance_mask
    )

    gt_ids = get_instance_ids(
        gt_instance_mask
    )

    candidate_matches = []

    for pred_id in pred_ids:

        pred_object = (
            pred_instance_mask == pred_id
        )

        for gt_id in gt_ids:

            gt_object = (
                gt_instance_mask == gt_id
            )

            iou = instance_iou(
                pred_object,
                gt_object
            )

            if iou >= iou_threshold:
                candidate_matches.append(
                    (iou, pred_id, gt_id)
                )

    candidate_matches.sort(
        reverse=True
    )

    matched_pred = set()
    matched_gt = set()

    matches = []

    for iou, pred_id, gt_id in candidate_matches:

        if pred_id in matched_pred:
            continue

        if gt_id in matched_gt:
            continue

        matched_pred.add(pred_id)
        matched_gt.add(gt_id)

        matches.append(
            (pred_id, gt_id, iou)
        )

    tp = len(matches)

    fp = len(pred_ids) - tp

    fn = len(gt_ids) - tp

    return tp, fp, fn, matches


def count_error(
    pred_instance_mask,
    gt_instance_mask
):
    """
    Erro absoluto no número de instâncias.
    """

    pred_ids = get_instance_ids(
        pred_instance_mask
    )

    gt_ids = get_instance_ids(
        gt_instance_mask
    )

    return abs(
        len(pred_ids) - len(gt_ids)
    )