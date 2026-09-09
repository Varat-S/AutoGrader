import os
import pytest
from app.models.grade import ColorGradeParams
from app.media.lut import generate_3d_cube_lut

def test_generate_3d_cube_lut(tmp_path):
    lut_file = str(tmp_path / 'test_grade.cube')
    params = ColorGradeParams(
        exposure_ev=0.5,
        contrast=1.1,
        saturation=1.2,
        lab_l_gain=1.1,
        lab_l_offset=5.0
    )
    
    output_path = generate_3d_cube_lut(params, lut_file, size=17)
    
    assert os.path.exists(output_path)
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Check headers
    assert any('LUT_3D_SIZE 17' in l for l in lines)
    assert any('DOMAIN_MIN' in l for l in lines)
    assert any('DOMAIN_MAX' in l for l in lines)
    
    # Check data rows (17^3 = 4913 data lines)
    data_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#') and not l.startswith('TITLE') and not l.startswith('LUT_3D_SIZE') and not l.startswith('DOMAIN')]
    assert len(data_lines) == 17 * 17 * 17
    
    # Check values normalized in [0.0, 1.0]
    for sample_line in data_lines[:10]:
        parts = list(map(float, sample_line.split()))
        assert len(parts) == 3
        for val in parts:
            assert 0.0 <= val <= 1.0

def test_shared_look_lut_equivalence_with_grade_plan(tmp_path):
    import numpy as np
    from app.models.analysis import CreativeSpecification, GlobalLookIntent
    from app.models.grade import GradePlan, CreativeLookParams
    from app.media.lut import generate_shared_creative_look_lut
    from app.media.color import apply_color_grade_to_frame

    spec = CreativeSpecification(
        look_title="Test Equivalence Look",
        target_aesthetic="Equivalence test aesthetic",
        contrast_intent=1.15,
        saturation_intent=1.10,
        highlight_bias="warm amber",
        shadow_bias="cool slate",
        highlight_rgb_offset=[0.02, 0.01, 0.05],
        shadow_rgb_offset=[-0.02, 0.0, -0.04],
        black_level_treatment="filmic lifted",
        black_mist_diffusion_strength=0.1
    )
    canonical = spec.get_canonical_global_look()

    lut_file = str(tmp_path / "shared_look.cube")
    size = 9
    generate_shared_creative_look_lut(spec, lut_file, size=size)

    # Parse LUT values: cube format is r_out g_out b_out ordered by b, then g, then r
    lut_values = []
    with open(lut_file, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l or l.startswith("#") or l.startswith("TITLE") or l.startswith("LUT_3D_SIZE") or l.startswith("DOMAIN"):
                continue
            parts = [float(x) for x in l.split()]
            lut_values.append(parts) # [r, g, b]

    # Evaluate directly with GradePlan on exact same lattice
    from app.tools.calculate_grade import parse_black_level_lift
    toe_lift = parse_black_level_lift(canonical.black_level_character, canonical.black_mist_diffusion_strength)
    plan = GradePlan(
        shot_id="shared_creative_look",
        is_same_scene=False,
        creative_look=CreativeLookParams(
            look_title=canonical.look_title,
            contrast=round(canonical.base_contrast, 3),
            pivot=0.45,
            saturation=round(canonical.base_saturation, 3),
            shadow_rgb_offset=canonical.shadow_rgb_offset,
            highlight_rgb_offset=canonical.highlight_rgb_offset,
            black_toe_lift=round(toe_lift, 2),
            black_mist_strength=canonical.black_mist_diffusion_strength
        )
    )

    r_space = np.linspace(0.0, 1.0, size, dtype=np.float32)
    g_space = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b_space = np.linspace(0.0, 1.0, size, dtype=np.float32)

    lattice = np.zeros((size * size, size, 3), dtype=np.float32)
    for b_idx in range(size):
        for g_idx in range(size):
            row_idx = b_idx * size + g_idx
            for r_idx in range(size):
                lattice[row_idx, r_idx] = [b_space[b_idx], g_space[g_idx], r_space[r_idx]]

    graded_lattice = apply_color_grade_to_frame(lattice, plan, is_log=False)

    max_diff = 0.0
    val_idx = 0
    for b_idx in range(size):
        for g_idx in range(size):
            row_idx = b_idx * size + g_idx
            for r_idx in range(size):
                b_direct, g_direct, r_direct = graded_lattice[row_idx, r_idx]
                r_lut, g_lut, b_lut = lut_values[val_idx]
                val_idx += 1

                diff_r = abs(float(r_direct) - r_lut)
                diff_g = abs(float(g_direct) - g_lut)
                diff_b = abs(float(b_direct) - b_lut)
                max_diff = max(max_diff, diff_r, diff_g, diff_b)

    assert max_diff < 1e-4, f"LUT output diverged from GradePlan by {max_diff}"
