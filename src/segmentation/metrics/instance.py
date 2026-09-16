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

def precision_recall(tp, fp, fn):
    """
    Calcula precision e recall.
    """

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    return precision, recall


def f1_score(tp, fp, fn):
    """
    Calcula o F1-score a partir de TP, FP e FN.
    """

    precision, recall = precision_recall(
        tp,
        fp,
        fn
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )

def evaluate_iou_thresholds(
    pred_instance_mask,
    gt_instance_mask
):
    """
    Avalia a segmentação em thresholds
    de IoU de 0.50 até 0.95, em passos de 0.05.
    """

    thresholds = np.arange(
        0.50,
        0.951,
        0.05
    )

    results = {}

    for threshold in thresholds:

        tp, fp, fn, matches = match_instances(
            pred_instance_mask,
            gt_instance_mask,
            iou_threshold=threshold
        )

        precision, recall = precision_recall(
            tp,
            fp,
            fn
        )

        f1 = f1_score(
            tp,
            fp,
            fn
        )

        results[round(threshold, 2)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matches": matches
        }

    return results

def mean_f1_over_thresholds(results):
    """
    Calcula a média do F1 nos thresholds avaliados.
    """

    values = [
        result["f1"]
        for result in results.values()
    ]

    if len(values) == 0:
        return 0.0

    return sum(values) / len(values)


def instance_confidences(pred_instance_mask, foreground_probability):
    """
    Calcula uma confiança para cada instância prevista.

    A confiança é a média da probabilidade de foreground
    nos pixels pertencentes àquela instância.

    Retorna uma lista de tuplas:

        (instance_id, confidence)
    """

    pred_ids = get_instance_ids(pred_instance_mask)

    confidences = []

    for pred_id in pred_ids:

        instance_pixels = (
            pred_instance_mask == pred_id
        )

        confidence = foreground_probability[
            instance_pixels
        ].mean()

        confidences.append(
            (pred_id, float(confidence))
        )

    return confidences


def match_predictions_for_ap(
    pred_instance_mask,
    gt_instance_mask,
    foreground_probability,
    iou_threshold=0.5
):
    """
    Faz o matching entre predições e ground truth
    respeitando a ordem de confiança.

    Retorna:
        tp: lista indicando TP para cada predição
        fp: lista indicando FP para cada predição
        scores: confiança de cada predição
    """

    pred_ids = get_instance_ids(pred_instance_mask)
    gt_ids = get_instance_ids(gt_instance_mask)

    confidences = instance_confidences(
        pred_instance_mask,
        foreground_probability
    )

    # Ordenar predições da maior para a menor confiança
    confidences.sort(
        key=lambda x: x[1],
        reverse=True
    )

    matched_gt = set()

    tp = []
    fp = []
    scores = []

    for pred_id, score in confidences:

        best_iou = 0.0
        best_gt_id = None

        pred_object = (
            pred_instance_mask == pred_id
        )

        for gt_id in gt_ids:

            if gt_id in matched_gt:
                continue

            gt_object = (
                gt_instance_mask == gt_id
            )

            iou = instance_iou(
                pred_object,
                gt_object
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        scores.append(score)

        if best_iou >= iou_threshold:

            tp.append(1)
            fp.append(0)

            matched_gt.add(best_gt_id)

        else:

            tp.append(0)
            fp.append(1)

    return (
        np.array(tp),
        np.array(fp),
        np.array(scores)
    )

def average_precision(
    pred_instance_mask,
    gt_instance_mask,
    foreground_probability,
    iou_threshold=0.5
):
    """
    Calcula Average Precision (AP) para um determinado
    threshold de IoU.
    """

    tp, fp, scores = match_predictions_for_ap(
        pred_instance_mask,
        gt_instance_mask,
        foreground_probability,
        iou_threshold
    )

    num_gt = len(
        get_instance_ids(gt_instance_mask)
    )

    if num_gt == 0:

        if len(tp) == 0:
            return 1.0

        return 0.0

    if len(tp) == 0:
        return 0.0

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    precision = (
        cumulative_tp
        / (
            cumulative_tp
            + cumulative_fp
        )
    )

    recall = (
        cumulative_tp
        / num_gt
    )

    # Adicionamos os extremos da curva
    mrec = np.concatenate(
        ([0.0], recall, [1.0])
    )

    mpre = np.concatenate(
        ([0.0], precision, [0.0])
    )

    # Precision envelope
    for i in range(
        len(mpre) - 2,
        -1,
        -1
    ):
        mpre[i] = max(
            mpre[i],
            mpre[i + 1]
        )

    # Pontos onde o recall muda
    indices = np.where(
        mrec[1:] != mrec[:-1]
    )[0]

    ap = np.sum(
        (
            mrec[indices + 1]
            - mrec[indices]
        )
        * mpre[indices + 1]
    )

    return float(ap)


def mean_average_precision(
    pred_instance_mask,
    gt_instance_mask,
    foreground_probability
):
    """
    Calcula AP para IoU de 0.50 até 0.95,
    com passo de 0.05.

    Retorna:
        ap_results: dicionário com AP para cada threshold
        map_value: média dos APs
    """

    thresholds = np.arange(
        0.50,
        0.951,
        0.05
    )

    ap_results = {}

    for threshold in thresholds:

        ap = average_precision(
            pred_instance_mask,
            gt_instance_mask,
            foreground_probability,
            iou_threshold=threshold
        )

        ap_results[
            round(threshold, 2)
        ] = ap

    map_value = np.mean(
        list(ap_results.values())
    )

    return ap_results, float(map_value)