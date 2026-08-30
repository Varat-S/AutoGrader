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
