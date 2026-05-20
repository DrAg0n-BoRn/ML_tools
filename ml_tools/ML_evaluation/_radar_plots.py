import plotly.graph_objects as go
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List, Union

from ..keys._keys import _EvaluationConfig


DEFAULT_PLOT_WIDTH = _EvaluationConfig.RADAR_PLOT_WIDTH
DEFAULT_PLOT_HEIGHT = _EvaluationConfig.RADAR_PLOT_HEIGHT


def mpl_to_plotly_rgba(color: str, alpha: float) -> str:
    r, g, b, _ = mcolors.to_rgba(color)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"

def calculate_smart_font_size(num_features: int, base_font_size: int, font_shrink_constant: int = 60) -> int:
    return max(base_font_size // 2, int(base_font_size * (font_shrink_constant / (font_shrink_constant + num_features))))

def calculate_smart_margin_left_right(max_feature_length: int) -> int:
    return int(max((DEFAULT_PLOT_WIDTH * 1.5 / 10), (DEFAULT_PLOT_WIDTH / 10) + (max_feature_length * DEFAULT_PLOT_WIDTH / 100)))


def save_radar_chart(
    scores: List[float], 
    target_names: List[str], 
    line_color: str, 
    fill_rgba: str, 
    title: str, 
    save_path_base: Union[str, Path], 
    margin_lr: int, 
    font_size: int, 
    tick_range: List[float] = [0.0, 1.0], 
    tick_vals: Optional[List[float]] = None
) -> None:
    if tick_vals is None:
        tick_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    default_ytick_size = max(12, font_size)
    default_tick_angle = 45
    default_ticklabels_angle = 45
    default_top_bottom_margin = DEFAULT_PLOT_HEIGHT // 10
    default_title_fontsize = font_size + 2

    fig = go.Figure(data=go.Scatterpolar(
        r=scores + [scores[0]],
        theta=target_names + [target_names[0]],
        fill='toself',
        fillcolor=fill_rgba, 
        line=dict(color=line_color, width=2.5),
        name=title
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=default_title_fontsize), x=0.5, xanchor='center'),
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=tick_range,
                tickvals=tick_vals, 
                angle=default_tick_angle, 
                tickangle=default_ticklabels_angle,
                tickfont=dict(size=default_ytick_size, color='dimgrey')
            ),
            angularaxis=dict(tickfont=dict(size=font_size))
        ),
        showlegend=False,
        height=DEFAULT_PLOT_HEIGHT,
        width=DEFAULT_PLOT_WIDTH,
        margin=dict(t=default_top_bottom_margin, b=default_top_bottom_margin, l=margin_lr, r=margin_lr),
    )
    
    fig.write_image(f"{save_path_base}.svg", format="svg")
    fig.write_html(f"{save_path_base}.html", include_plotlyjs=True)
