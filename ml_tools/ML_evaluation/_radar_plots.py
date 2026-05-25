import textwrap
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List, Union

from ..keys._keys import _EvaluationConfig
from .._core import get_logger

from ._helpers import wrap_text


_LOGGER = get_logger("Radar Plot")


DEFAULT_PLOT_WIDTH = _EvaluationConfig.RADAR_PLOT_WIDTH
DEFAULT_PLOT_HEIGHT = _EvaluationConfig.RADAR_PLOT_HEIGHT
MAX_FEATURES_BEFORE_DYNAMIC_SIZING = _EvaluationConfig.RADAR_MAX_FEATURES_BEFORE_DYNAMIC_SIZING
MAX_FEATURE_NAME_LENGTH_FOR_MARGIN = _EvaluationConfig.RADAR_MAX_FEATURE_NAME_LENGTH_FOR_MARGIN


def mpl_to_plotly_rgba(color: str, alpha: float) -> str:
    r, g, b, _ = mcolors.to_rgba(color)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"

def calculate_smart_font_size(num_features: int, base_font_size: int, font_shrink_constant: int = 60) -> int:
    return max(base_font_size // 2, int(base_font_size * (font_shrink_constant / (font_shrink_constant + num_features))))

def calculate_smart_margin_left_right(max_feature_length: int) -> int:
    # Since text is now wrapped at N characters, the horizontal space requirement is naturally capped.
    effective_length = min(max_feature_length, MAX_FEATURE_NAME_LENGTH_FOR_MARGIN)
    calculated_margin = (DEFAULT_PLOT_WIDTH / 10) + (effective_length * DEFAULT_PLOT_WIDTH / 100)
    
    # Cap the margin at 25% of the total width so the plot area remains dominant
    max_allowed = DEFAULT_PLOT_WIDTH * 0.25
    return int(max(DEFAULT_PLOT_WIDTH * 0.15, min(calculated_margin, max_allowed)))


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
    default_tick_angle = 45 + 90 # top-left quadrant
    default_ticklabels_angle = 45 + 90 # top-left quadrant
    default_title_fontsize = font_size + 2
    
    # 1. Dynamic Canvas Size: Expand the base size if there are more than N features
    num_feats = len(target_names)
    dynamic_width = DEFAULT_PLOT_WIDTH if num_feats <= MAX_FEATURES_BEFORE_DYNAMIC_SIZING else int(DEFAULT_PLOT_WIDTH * (1 + (num_feats - MAX_FEATURES_BEFORE_DYNAMIC_SIZING) * 0.02))
    dynamic_height = DEFAULT_PLOT_HEIGHT if num_feats <= MAX_FEATURES_BEFORE_DYNAMIC_SIZING else int(DEFAULT_PLOT_HEIGHT * (1 + (num_feats - MAX_FEATURES_BEFORE_DYNAMIC_SIZING) * 0.02))

    # 2. Adjust top/bottom margins to accommodate multi-line wrapped text at the poles
    default_top_margin = int(dynamic_height * 0.15)
    default_bottom_margin = int(dynamic_height * 0.15)

    # 3. Smart Text Wrapping: Replace underscores with spaces, wrap at N chars, use <br> for Plotly
    str_target_names = []
    for name in target_names:
        wrapped_name = wrap_text(name, width=MAX_FEATURE_NAME_LENGTH_FOR_MARGIN, break_char="<br>")
        str_target_names.append(f"{wrapped_name}\u200b")

    # 4. Base figure
    fig = go.Figure(data=go.Scatterpolar(
        r=scores + [scores[0]],
        theta=str_target_names + [str_target_names[0]],
        fill='toself',
        fillcolor=fill_rgba, 
        line=dict(color=line_color, width=2.5),
        name=title
    ))
    
    # 5. Interactive Font slider
    # y-coordinates below 0 place the sliders underneath the chart
    slider_y_positions = [-0.20, -0.35, -0.50]
    slider_labels = ["Title Font", "Feature Names Font", "Y-Tick Font"]
    slider_args = [
        ["title.font.size", "polar.angularaxis.tickfont.size", "polar.radialaxis.tickfont.size"],
        [default_title_fontsize, font_size, default_ytick_size]
    ]

    sliders = []
    for i, (path, base_size) in enumerate(zip(slider_args[0], slider_args[1])):
        min_f, max_f = max(6, base_size // 2), base_size * 3
        steps = [
            dict(method="relayout", args=[{path: s}], label=str(s))
            for s in range(min_f, max_f + 1)
        ]
        
        sliders.append(dict(
            active=base_size,
            currentvalue={"prefix": f"{slider_labels[i]}: ", "font": {"size": 12}}, # Smaller title font
            pad={"t": 5, "b": 5}, # Tighter padding
            x=0.25, y=slider_y_positions[i], len=1, # Center the sliders and make them 100% width
            xanchor="left", yanchor="top",
            font=dict(size=10), # Smaller slider number font
            steps=steps
        ))

    # 5. Base Layout Update (Without Sliders)    
    fig.update_layout(
        title=dict(text=title, font=dict(size=default_title_fontsize), x=0.5, xanchor='center'),
        polar=dict(
            radialaxis=dict(
                visible=True, range=tick_range, tickvals=tick_vals,
                angle=default_tick_angle, tickangle=default_ticklabels_angle,
                tickfont=dict(size=default_ytick_size, color='dimgrey'),
                showline=False
            ),
            angularaxis=dict(
                type='category',
                categoryarray=str_target_names,
                tickfont=dict(size=font_size)
            )
        ),
        showlegend=False,
        height=dynamic_height,
        width=dynamic_width,
        margin=dict(t=default_top_margin, b=default_bottom_margin, l=margin_lr, r=margin_lr),
    )
    
    # Save the clean static SVG
    try:
        fig.write_image(f"{save_path_base}.svg", format="svg")
    except Exception as e:
        _LOGGER.error(f"Failed to save SVG radar chart due to: {e}")

    # 6. Inject Interactive Sliders and Save HTML
    # Add absolute height and massive bottom margin exclusively for the HTML to house the spaced-out sliders
    html_dynamic_height = dynamic_height + int(dynamic_height * 0.40)
    html_bottom_margin = int(dynamic_height * 0.60)
    
    fig.update_layout(
        sliders=sliders,
        height=html_dynamic_height,
        margin=dict(t=default_top_margin, b=html_bottom_margin, l=margin_lr, r=margin_lr)
    )
    
    # Configure the HTML modebar to download as SVG by default
    html_config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': Path(save_path_base).name
        }
    }
    
    # Check for chromium based browser availability
    try:
        fig.write_html(f"{save_path_base}.html", include_plotlyjs='cdn', config=html_config)
    except Exception as e:
        _LOGGER.error(f"Failed to save interactive HTML radar chart due to: {e}")
        
