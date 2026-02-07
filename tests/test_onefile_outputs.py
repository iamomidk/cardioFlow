from pathlib import Path

from cardio_onefile import PF1Params, plot_onepage_comparison, simulate_af, simulate_healthy


def test_outputs_figure_exists_non_empty(tmp_path: Path):
    params = PF1Params()
    healthy = simulate_healthy(params, tf=1.0, dt_out=0.002)
    af = simulate_af(params, tf=1.0, dt_out=0.002, seed=1)
    fig = plot_onepage_comparison(healthy, af, t_start_s=0.0, t_end_s=1.0)

    output_pdf = tmp_path / "comparison_onepage.pdf"
    fig.savefig(output_pdf)

    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0
