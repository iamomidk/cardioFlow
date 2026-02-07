import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from cardio_onefile import PF1Params, export_onepage, make_healthy_vs_af_figure, simulate_af, simulate_healthy


def test_onepage_figure_integrity(tmp_path: Path) -> None:
    params = PF1Params()
    healthy = simulate_healthy(params, tf=3.0, dt_out=0.002)
    af = simulate_af(params, tf=3.0, dt_out=0.002, seed=1)

    fig = make_healthy_vs_af_figure(healthy, af, t_start=0.0, t_end=3.0)
    assert len(fig.axes) == 5  # 4 panels + 1 twin axis

    # Panel (a) ≥4 lines
    assert len(fig.axes[0].lines) >= 4
    # Panel (b) ≥2 lines
    assert len(fig.axes[1].lines) >= 2
    # Panel (c) ≥6 lines across both axes
    assert len(fig.axes[2].lines) + len(fig.axes[3].lines) >= 6
    # Panel (d) ≥2 lines
    assert len(fig.axes[4].lines) >= 2

    outpath = tmp_path / "comparison_onepage.pdf"
    export_onepage(fig, str(outpath))
    assert outpath.exists()
    assert outpath.stat().st_size > 1000
