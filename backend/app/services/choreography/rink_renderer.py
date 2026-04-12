"""SVG rink diagram renderer.

Generates a top-down orthographic view of a 60m x 30m ice rink
with element markers, labels, and connecting paths.
"""

from __future__ import annotations


def render_rink(
    elements: list[dict],
    *,
    width: int = 1200,
    height: int = 600,
) -> str:
    """Render a rink diagram as SVG string.

    Args:
        elements: list of dicts with "code", "position" ({x, y}), "timestamp".
        width: SVG width in pixels.
        height: SVG height in pixels.

    Returns:
        SVG string.
    """
    rink_w, rink_h = 60.0, 30.0
    parts: list[str] = []

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 60 30">'
    )
    parts.append('<rect x="0" y="0" width="60" height="30" fill="#e8f0fe" rx="1"/>')
    parts.append(
        '<rect x="1" y="1" width="58" height="28" fill="none" stroke="#2563eb" stroke-width="0.15" rx="0.5"/>'
    )
    parts.append(
        '<line x1="30" y1="1" x2="30" y2="29" stroke="#dc2626" stroke-width="0.1" stroke-dasharray="0.5,0.5"/>'
    )
    parts.append(
        '<circle cx="30" cy="15" r="4.5" fill="none" stroke="#dc2626" stroke-width="0.1"/>'
    )
    parts.append('<circle cx="30" cy="15" r="0.15" fill="#dc2626"/>')
    parts.append('<line x1="5" y1="1" x2="5" y2="29" stroke="#2563eb" stroke-width="0.08"/>')
    parts.append('<line x1="55" y1="1" x2="55" y2="29" stroke="#2563eb" stroke-width="0.08"/>')

    for cx, cy in [(10, 7.5), (10, 22.5), (50, 7.5), (50, 22.5)]:
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="3" fill="none" stroke="#2563eb" stroke-width="0.08"/>'
        )
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="0.15" fill="#dc2626"/>')

    for i, el in enumerate(elements):
        pos = el.get("position")
        if not pos:
            continue
        x, y = pos["x"], pos["y"]
        code = el.get("code", "")

        is_spin = "Sp" in code
        is_step = "StSq" in code
        is_choreo = "ChSq" in code

        if is_spin:
            parts.append(
                f'<circle cx="{x}" cy="{y}" r="1.2" fill="#9333ea" opacity="0.3" stroke="#9333ea" stroke-width="0.1"/>'
            )
            color = "#9333ea"
        elif is_step:
            parts.append(
                f'<rect x="{x - 1}" y="{y - 0.5}" width="2" height="1" fill="none" stroke="#16a34a" stroke-width="0.1" stroke-dasharray="0.3,0.2"/>'
            )
            color = "#16a34a"
        elif is_choreo:
            parts.append(
                f'<polygon points="{x},{y - 0.8} {x + 0.8},{y} {x},{y + 0.8} {x - 0.8},{y}" fill="#2563eb" opacity="0.3" stroke="#2563eb" stroke-width="0.1"/>'
            )
            color = "#2563eb"
        else:
            parts.append(f'<circle cx="{x}" cy="{y}" r="0.6" fill="#ea580c" opacity="0.8"/>')
            color = "#ea580c"

        parts.append(
            f'<text x="{x}" y="{y - 1.2}" text-anchor="middle" font-size="1.2" fill="{color}" font-weight="bold">{code}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{y + 0.3}" text-anchor="middle" font-size="0.7" fill="#666">{i + 1}</text>'
        )

        if i < len(elements) - 1:
            next_pos = elements[i + 1].get("position")
            if next_pos:
                parts.append(
                    f'<line x1="{x}" y1="{y}" x2="{next_pos["x"]}" y2="{next_pos["y"]}" '
                    f'stroke="#94a3b8" stroke-width="0.06" stroke-dasharray="0.3,0.2" opacity="0.6"/>'
                )

    parts.append("</svg>")
    return "\n".join(parts)
