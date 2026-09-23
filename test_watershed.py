import numpy as np

from watershed import watershed_from_probabilities


def create_touching_objects(
    height=128,
    width=128
):

    yy, xx = np.ogrid[
        :height,
        :width
    ]

    center_y = height // 2

    center_x_1 = 48
    center_x_2 = 80

    radius = 20

    object_1 = (
        (xx - center_x_1) ** 2
        + (yy - center_y) ** 2
        <= radius ** 2
    )

    object_2 = (
        (xx - center_x_2) ** 2
        + (yy - center_y) ** 2
        <= radius ** 2
    )

    foreground = (
        object_1 | object_2
    )

    interior_1 = (
        (xx - center_x_1) ** 2
        + (yy - center_y) ** 2
        <= (radius - 5) ** 2
    )

    interior_2 = (
        (xx - center_x_2) ** 2
        + (yy - center_y) ** 2
        <= (radius - 5) ** 2
    )

    interior = (
        interior_1 | interior_2
    )

    boundary = (
        foreground
        & ~interior
    )

    return (
        foreground,
        interior,
        boundary
    )


def test_two_touching_objects():

    (
        foreground,
        interior,
        boundary
    ) = create_touching_objects()

    height, width = foreground.shape

    probabilities = np.zeros(
        (3, height, width),
        dtype=np.float32
    )

    probabilities[0] = (
        1.0 - foreground.astype(np.float32)
    )

    probabilities[1] = (
        interior.astype(np.float32)
    )

    probabilities[2] = (
        boundary.astype(np.float32)
    )

    (
        instance_mask,
        markers,
        foreground_mask
    ) = watershed_from_probabilities(
        probabilities,
        interior_threshold=0.5,
        foreground_threshold=0.5,
        min_marker_size=10
    )

    number_of_markers = len(
        np.unique(markers)
    ) - 1

    number_of_instances = len(
        np.unique(instance_mask)
    ) - 1

    assert number_of_markers == 2, (
        "O teste deveria produzir "
        "exatamente dois marcadores."
    )

    assert number_of_instances == 2, (
        "O watershed deveria separar "
        "os dois objetos em duas instâncias."
    )

    assert np.array_equal(
        foreground_mask,
        foreground
    ), ("O foreground previsto deveria corresponder ao foreground sintético.")

    print("Teste do watershed concluído com sucesso.")

    print(f"Marcadores: {number_of_markers}")

    print(f"Instâncias: {number_of_instances}")


if __name__ == "__main__":
    test_two_touching_objects()