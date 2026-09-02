import pytest
from pathlib import Path
import numpy as np
from app.models.grade import GradePlan, CreativeLookParams
from app.media.color import apply_color_grade_to_frame
from app.media.lut import generate_3d_cube_lut

def test_highlight_bias_modifies_only_highlights():
    # Create dark shadow patch (L ~ 20) and bright highlight patch (L ~ 85)
    dark_patch = np.full((50, 50, 3), 35, dtype=np.uint8)
    bright_patch = np.full((50, 50, 3), 215, dtype=np.uint8)
    
    # Plan with warm amber highlight bias (+R, -B) and NO shadow bias
    plan = GradePlan(shot_id="hl_test")
    plan.creative_look.highlight_rgb_offset = [-0.05, 0.01, 0.06] # [B, G, R]
    plan.creative_look.shadow_rgb_offset = [0.0, 0.0, 0.0]
    
    graded_dark = apply_color_grade_to_frame(dark_patch, plan)
    graded_bright = apply_color_grade_to_frame(bright_patch, plan)
    
    # Highlight patch must show warm tint: Red > Blue
    assert int(graded_bright[0, 0, 2]) > int(graded_bright[0, 0, 0]) + 10, "Highlights must receive warm amber bias"
    
    # Shadow patch must NOT have received highlight bias (neutral B == G == R within 1 value)
    assert abs(int(graded_dark[0, 0, 2]) - int(graded_dark[0, 0, 0])) <= 1, "Shadows must remain untouched by highlight bias"

def test_shadow_bias_modifies_only_shadows():
    dark_patch = np.full((50, 50, 3), 35, dtype=np.uint8)
    bright_patch = np.full((50, 50, 3), 215, dtype=np.uint8)
    
    # Plan with cool slate shadow bias (+B, -R) and NO highlight bias
    plan = GradePlan(shot_id="sh_test")
    plan.creative_look.shadow_rgb_offset = [0.06, 0.01, -0.04] # [B, G, R]
    plan.creative_look.highlight_rgb_offset = [0.0, 0.0, 0.0]
    
    graded_dark = apply_color_grade_to_frame(dark_patch, plan)
    graded_bright = apply_color_grade_to_frame(bright_patch, plan)
    
    # Shadow patch must show cool slate tint: Blue > Red
    assert int(graded_dark[0, 0, 0]) > int(graded_dark[0, 0, 2]) + 5, "Shadows must receive cool slate bias"
    
    # Bright patch must NOT have received shadow bias
    assert abs(int(graded_bright[0, 0, 2]) - int(graded_bright[0, 0, 0])) <= 1, "Highlights must remain untouched by shadow bias"

def test_float32_cube_lut_generation(tmp_path):
    plan = GradePlan(shot_id="lut_test")
    plan.creative_look.contrast = 1.15
    plan.creative_look.saturation = 1.10
    plan.creative_look.highlight_rgb_offset = [-0.03, 0.0, 0.04]
    
    lut_file = str(tmp_path / "test_grade.cube")
    generate_3d_cube_lut(plan, lut_file, size=17, title="Test_Float_LUT")
    
    assert Path(lut_file).exists()
    with open(lut_file, "r") as f:
        content = f.read()
        
    assert "TITLE \"Test_Float_LUT\"" in content
    assert "LUT_3D_SIZE 17" in content
    assert "DOMAIN_MIN 0.0 0.0 0.0" in content
    assert "DOMAIN_MAX 1.0 1.0 1.0" in content

def test_lut_continuous_float_precision_not_quantized_to_uint8(tmp_path):
    plan = GradePlan(shot_id="lut_precision_test")
    plan.creative_look.contrast = 1.15
    plan.creative_look.highlight_rgb_offset = [-0.035, 0.012, 0.048]
    plan.creative_look.shadow_rgb_offset = [0.045, 0.008, -0.025]
    
    lut_file = str(tmp_path / "test_precision.cube")
    generate_3d_cube_lut(plan, lut_file, size=33)
    
    vals = []
    with open(lut_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and not line.startswith("#") and not line.startswith("LUT") and not line.startswith("TITLE") and not line.startswith("DOMAIN"):
                try:
                    vals.extend([float(p) for p in parts])
                except ValueError:
                    pass
                    
    vals = np.array(vals)
    # Compute distance of (val * 255) from nearest integer
    uint8_dist = np.abs((vals * 255.0) - np.round(vals * 255.0))
    mean_quant_dist = np.mean(uint8_dist)
    assert mean_quant_dist > 0.05, f"LUT values must be continuous floats, not quantized to 1/255 steps! Got {mean_quant_dist}"