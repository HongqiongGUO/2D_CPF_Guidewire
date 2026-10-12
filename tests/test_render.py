"""Both contact-storage layouts render their last numerical endpoint."""

from dataclasses import replace
import pytest
from cpf2d.config import default_scenario
from cpf2d.render import render_run
from cpf2d.simulation import run


@pytest.mark.parametrize("case", ["free", "insertion"])
def test_last_endpoint_png(case, output_dir):
    run(replace(default_scenario(case), duration=0.001), output_dir)
    target = render_run(output_dir)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
