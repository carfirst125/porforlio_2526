from app.core.color_utils import hex_to_hsl, tint_shade_from_dominant


def test_tint_and_shade_keep_dominant_hue_and_move_lightness_only_slightly():
    dominant = hex_to_hsl("#0F472F")
    tint_hex, shade_hex = tint_shade_from_dominant("#0F472F")
    tint = hex_to_hsl(tint_hex)
    shade = hex_to_hsl(shade_hex)

    assert dominant is not None and tint is not None and shade is not None
    assert abs(tint[0] - dominant[0]) < 0.01
    assert abs(shade[0] - dominant[0]) < 0.01
    assert tint[1] > dominant[1] > shade[1]


def test_invalid_dominant_color_uses_safe_defaults():
    assert tint_shade_from_dominant("not-a-color") == ("#F5F5F5", "#4D4D4D")
