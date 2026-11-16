from app.core.canvas_planner import (
    GEMINI_31_FLASH_IMAGE_RATIOS,
    content_safe_box,
    plan_canvas_for_output,
    plan_request_geometry,
)
from app.core.pipeline import load_request_json


def test_ultrawide_output_uses_nearest_native_ratio_with_small_center_crop():
    plan = plan_canvas_for_output((1035, 420), GEMINI_31_FLASH_IMAGE_RATIOS)

    assert plan.ratio_label == "21:9"
    assert plan.canvas_wh == (1035, 444)
    assert plan.crop_box == (0, 12, 1035, 420)
    assert plan.crop_loss < 0.06


def test_exact_supported_ratio_needs_no_crop():
    plan = plan_canvas_for_output((1536, 192), GEMINI_31_FLASH_IMAGE_RATIOS)

    assert plan.ratio_label == "8:1"
    assert plan.canvas_wh == (1536, 192)
    assert plan.crop_box == (0, 0, 1536, 192)
    assert not plan.has_crop


def test_square_and_portrait_sizes_are_not_hardcoded_to_banner_geometry():
    square = plan_canvas_for_output((1080, 1080), GEMINI_31_FLASH_IMAGE_RATIOS)
    portrait = plan_canvas_for_output((1080, 1350), GEMINI_31_FLASH_IMAGE_RATIOS)

    assert square.ratio_label == "1:1"
    assert square.canvas_wh == (1080, 1080)
    assert portrait.ratio_label == "4:5"
    assert portrait.canvas_wh == (1080, 1350)


def test_request_copy_uses_output_as_source_of_truth_without_editing_json():
    original = load_request_json("DB_NOTI_DEMO_TRUNGTHU_huy")
    planned, plan = plan_request_geometry(original, ["gemini-3.1-flash-image"])
    expected = plan_canvas_for_output(original.final_output_wh(), GEMINI_31_FLASH_IMAGE_RATIOS)

    assert plan is not None
    assert plan == expected
    assert planned.canvas_wh() == expected.canvas_wh
    assert planned.crop_box() == expected.crop_box
    assert planned.final_output_wh() == original.final_output_wh()


def test_unknown_or_mixed_model_preserves_declared_geometry():
    original = load_request_json("DB_NOTI_DEMO_TRUNGTHU_huy")
    planned, plan = plan_request_geometry(original, ["gemini-3.1-flash-image", "gpt-image-2"])

    assert plan is None
    assert planned is original


def test_content_safe_box_is_proportional_and_size_independent():
    assert content_safe_box((82, 0, 1035, 150)) == (134, 8, 931, 134)
    assert content_safe_box((0, 18, 1035, 222)) == (52, 29, 931, 200)
