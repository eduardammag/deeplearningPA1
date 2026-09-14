import numpy as np


# ============================================================
# IoU
# ============================================================

def mask_iou(
    mask_a,
    mask_b
):
    """
    Calcula IoU entre duas máscaras booleanas.
    """

    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:

        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
# União-Find
# ============================================================

class UnionFind:

    def __init__(
        self,
        values
    ):

        self.parent = {
            value: value
            for value in values
        }

    def find(
        self,
        value
    ):

        if self.parent[value] != value:

            self.parent[value] = self.find(
                self.parent[value]
            )

        return self.parent[value]

    def union(
        self,
        value_a,
        value_b
    ):

        root_a = self.find(
            value_a
        )

        root_b = self.find(
            value_b
        )

        if root_a != root_b:

            self.parent[root_b] = root_a


# ============================================================
# Região de sobreposição
# ============================================================

def overlap_region(
    position_a,
    position_b
):
    """
    Calcula a região espacial de sobreposição entre dois tiles.

    Retorna:

        (y_start, y_end, x_start, x_end)

    ou None se os tiles não se sobrepõem.
    """

    (
        y1_start,
        y1_end,
        x1_start,
        x1_end
    ) = position_a

    (
        y2_start,
        y2_end,
        x2_start,
        x2_end
    ) = position_b

    y_start = max(
        y1_start,
        y2_start
    )

    y_end = min(
        y1_end,
        y2_end
    )

    x_start = max(
        x1_start,
        x2_start
    )

    x_end = min(
        x1_end,
        x2_end
    )

    if (
        y_start >= y_end
        or
        x_start >= x_end
    ):

        return None

    return (
        y_start,
        y_end,
        x_start,
        x_end
    )


# ============================================================
# Instâncias globais dos tiles
# ============================================================

def build_global_tile_instances(
    tile_predictions
):
    """
    Converte as máscaras locais dos tiles em máscaras globais.

    Cada instância recebe um ID global único.

    Retorna:

        instances

    Cada item contém:

        global_id
        tile_index
        position
        mask
    """

    instances = []

    for tile in tile_predictions:

        (
            y_start,
            y_end,
            x_start,
            x_end
        ) = tile["position"]

        local_mask = tile[
            "instance_mask"
        ]

        mapping = tile[
            "local_to_global"
        ]

        for local_id, global_id in mapping.items():

            local_instance = (
                local_mask ==
                local_id
            )

            instances.append(
                {
                    "global_id": global_id,

                    "tile_index":
                        tile["tile_index"],

                    "position":
                        tile["position"],

                    "local_mask":
                        local_instance
                }
            )

    return instances


# ============================================================
# Comparação entre instâncias na sobreposição
# ============================================================

def find_merge_candidates(
    instances,
    iou_threshold=0.30
):
    """
    Procura pares de instâncias que provavelmente representam
    o mesmo objeto em tiles diferentes.

    O IoU é calculado somente na região de sobreposição
    espacial dos dois tiles.
    """

    candidates = []

    for index_a in range(
        len(instances)
    ):

        instance_a = instances[
            index_a
        ]

        for index_b in range(
            index_a + 1,
            len(instances)
        ):

            instance_b = instances[
                index_b
            ]

            if (
                instance_a["tile_index"]
                ==
                instance_b["tile_index"]
            ):

                continue

            overlap = overlap_region(
                instance_a["position"],
                instance_b["position"]
            )

            if overlap is None:

                continue

            (
                y_start,
                y_end,
                x_start,
                x_end
            ) = overlap

            (
                ay_start,
                ay_end,
                ax_start,
                ax_end
            ) = instance_a[
                "position"
            ]

            (
                by_start,
                by_end,
                bx_start,
                bx_end
            ) = instance_b[
                "position"
            ]

            # ------------------------------------------------
            # Coordenadas relativas ao tile A
            # ------------------------------------------------

            a_y_start = (
                y_start -
                ay_start
            )

            a_y_end = (
                y_end -
                ay_start
            )

            a_x_start = (
                x_start -
                ax_start
            )

            a_x_end = (
                x_end -
                ax_start
            )

            # ------------------------------------------------
            # Coordenadas relativas ao tile B
            # ------------------------------------------------

            b_y_start = (
                y_start -
                by_start
            )

            b_y_end = (
                y_end -
                by_start
            )

            b_x_start = (
                x_start -
                bx_start
            )

            b_x_end = (
                x_end -
                bx_start
            )

            mask_a = instance_a[
                "local_mask"
            ][
                a_y_start:a_y_end,
                a_x_start:a_x_end
            ]

            mask_b = instance_b[
                "local_mask"
            ][
                b_y_start:b_y_end,
                b_x_start:b_x_end
            ]

            iou = mask_iou(
                mask_a,
                mask_b
            )

            if iou >= iou_threshold:

                candidates.append(
                    {
                        "global_id_a":
                            instance_a["global_id"],

                        "global_id_b":
                            instance_b["global_id"],

                        "iou":
                            float(iou)
                    }
                )

    candidates.sort(
        key=lambda item: item["iou"],
        reverse=True
    )

    return candidates


# ============================================================
# Fusão
# ============================================================

def fuse_instances(
    tiled_result,
    iou_threshold=0.30
):
    """
    Funde instâncias provenientes de tiles sobrepostos.

    A fusão é feita por Union-Find sobre os pares cuja IoU
    na região de sobreposição supera o limiar.

    Retorna:

        fused_instance_mask
        merge_candidates
        groups
    """

    tile_predictions = tiled_result[
        "tile_predictions"
    ]

    instances = build_global_tile_instances(
        tile_predictions
    )

    if len(instances) == 0:

        height = tiled_result[
            "naive_instance_mask"
        ].shape[0]

        width = tiled_result[
            "naive_instance_mask"
        ].shape[1]

        return (
            np.zeros(
                (height, width),
                dtype=np.int32
            ),
            [],
            []
        )

    ids = [
        instance["global_id"]
        for instance in instances
    ]

    union_find = UnionFind(
        ids
    )

    candidates = find_merge_candidates(
        instances,
        iou_threshold=iou_threshold
    )

    for candidate in candidates:

        union_find.union(
            candidate["global_id_a"],
            candidate["global_id_b"]
        )

    groups = {}

    for global_id in ids:

        root = union_find.find(
            global_id
        )

        if root not in groups:

            groups[root] = []

        groups[root].append(
            global_id
        )

    # --------------------------------------------------------
    # Criar máscara final
    # --------------------------------------------------------

    naive_mask = tiled_result[
        "naive_instance_mask"
    ]

    fused_mask = np.zeros_like(
        naive_mask,
        dtype=np.int32
    )

    root_to_new_id = {}

    next_id = 1

    for root in sorted(
        groups.keys()
    ):

        root_to_new_id[root] = (
            next_id
        )

        next_id += 1

    # --------------------------------------------------------
    # Para cada pixel da máscara ingênua, descobrir a que
    # grupo sua instância pertence.
    # --------------------------------------------------------

    id_to_root = {}

    for global_id in ids:

        id_to_root[
            global_id
        ] = union_find.find(
            global_id
        )

    for global_id, root in id_to_root.items():

        output_id = root_to_new_id[
            root
        ]

        mask = (
            naive_mask ==
            global_id
        )

        fused_mask[
            mask
        ] = output_id

    return (
        fused_mask,
        candidates,
        list(groups.values())
    )


# ============================================================
# Estatísticas
# ============================================================

def count_instances(
    instance_mask
):
    """
    Conta instâncias em uma máscara.
    """

    ids = np.unique(
        instance_mask
    )

    return int(
        np.sum(ids != 0)
    )


def fusion_summary(
    naive_mask,
    fused_mask,
    candidates
):
    """
    Produz resumo da fusão.
    """

    naive_count = count_instances(
        naive_mask
    )

    fused_count = count_instances(
        fused_mask
    )

    return {
        "naive_count":
            naive_count,

        "fused_count":
            fused_count,

        "num_merges":
            len(candidates),

        "reduction":
            naive_count - fused_count
    }


# ============================================================
# Teste simples
# ============================================================

def main():

    print(
        "Módulo de fusão de instâncias."
    )

    print(
        "A fusão é feita por IoU na "
        "região de sobreposição dos tiles."
    )


if __name__ == "__main__":

    main()