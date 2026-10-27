"""Dependency-free SVG line charts for exact finite-field counts.

The CSV is the numerical record. These linear-axis charts compare the
printed formula with independently counted results; no source pixels copied.
"""
from pathlib import Path


def line_chart(rows,series,title,path):
    width=740;height=430;left=105;right=25;top=60;bottom=65
    W=width-left-right;H=height-top-bottom
    xs=[r['p'] for r in rows];maximum=max(r[key] for r in rows for key,label,color in series)
    X=lambda x:left+W*(x-xs[0])/(xs[-1]-xs[0])
    Y=lambda y:top+H*(1-y/maximum)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="20" y="25" font-family="sans-serif" font-size="17">{title}</text>']
    for i,(key,label,color) in enumerate(series):
        out.append(f'<text x="{20+i*350}" y="46" font-family="sans-serif" font-size="12" fill="{color}">{label}</text>')
    for j in range(6):
        value=maximum*j/5;y=Y(value)
        out.append(f'<path d="M {left} {y:.3f} H {width-right}" stroke="#ddd"/>')
        out.append(f'<text x="{left-8}" y="{y+4:.3f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:,.0f}</text>')
    for p in xs:
        out.append(f'<text x="{X(p):.3f}" y="{top+H+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{p}</text>')
    out.append(f'<path d="M {left} {top} V {top+H} H {width-right}" fill="none" stroke="#333"/>')
    for key,label,color in series:
        points=' '.join(f'{X(r["p"]):.3f},{Y(r[key]):.3f}' for r in rows)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for r in rows:out.append(f'<circle cx="{X(r["p"]):.3f}" cy="{Y(r[key]):.3f}" r="3" fill="{color}"/>')
    out.append(f'<text x="{left+W/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="12">Prime p (linear axes); exact values in figure-data.csv</text></svg>')
    Path(path).write_text('\n'.join(out)+'\n')
