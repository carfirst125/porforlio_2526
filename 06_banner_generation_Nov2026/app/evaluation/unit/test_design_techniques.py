from app.core import design_techniques


def test_empty_technique_selection_is_noop():
    resolved, unknown = design_techniques.resolve_techniques([])
    assert resolved == []
    assert unknown == []


def test_known_and_unknown_techniques_resolve_safely():
    resolved, unknown = design_techniques.resolve_techniques(["radial_spotlight", "not_a_real_technique"])
    assert [item.key for item in resolved] == ["radial_spotlight"]
    assert resolved[0].prompt_instruction
    assert unknown == ["not_a_real_technique"]


def test_schema_v2_catalog_is_complete_and_referentially_valid():
    catalog = design_techniques.load_catalog()
    assert catalog.schema_version == "2.0"
    assert catalog.validation_errors == ()
    assert len(catalog.techniques) >= 40
    assert set(item.group for item in catalog.techniques.values()) == {"background", "decoration", "product"}
    assert catalog.selection_policy["min_per_group"] == 0
    assert catalog.selection_policy["max_per_group"] == 3
    assert "silhoutte_watermark" in catalog.techniques
    assert "silhouette_watermark" in catalog.techniques["silhoutte_watermark"].aliases

    keys = set(catalog.techniques)
    for item in catalog.techniques.values():
        assert item.label
        assert item.description
        assert item.prompt_instruction
        assert item.visual_density in {"low", "medium", "high"}
        assert set(item.incompatible_with) <= keys
        assert set(item.works_well_with) <= keys
        for conflict in item.incompatible_with:
            assert item.key in catalog.techniques[conflict].incompatible_with
