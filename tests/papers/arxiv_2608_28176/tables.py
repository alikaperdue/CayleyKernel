"""Paper tables transcribed independently of the product implementation.

Table 1 uses f0..f9, t0..t9 for f and tilde(f). A leading integer is
its coefficient. Source: printed page 16, visually checked against PDF.
"""
TABLE1 = '''
f0 f1 f2 f3 f4 f5 f6 t0 t1 t2 t3 t4 t5 t6
f1 -2f0 2f3 -2f2 0 0 2f7 t1 -2t0 -2t3 2t2 0 0 -2t7
f2 -2f3 -2f0 2f1 0 2f8 0 t2 2t3 -2t0 -2t1 0 -2t8 0
f3 2f2 -2f1 -2f0 2f9 0 0 t3 -2t2 2t1 -2t0 -2t9 0 0
f4 0 0 -2f9 -2f0 -2f6 2f5 t4 0 0 2t9 -2t0 2t6 -2t5
f5 0 -2f8 0 2f6 -2f0 -2f4 t5 0 2t8 0 -2t6 -2t0 2t4
f6 -2f7 0 0 -2f5 2f4 -2f0 t6 2t7 0 0 2t5 -2t4 -2t0
t0 -t1 -t2 -t3 -t4 -t5 -t6 -f0 f1 f2 f3 f4 f5 f6
t1 2t0 -2t3 2t2 0 0 -2t7 -f1 -2f0 -2f3 2f2 0 0 -2f7
t2 2t3 2t0 -2t1 0 -2t8 0 -f2 2f3 -2f0 -2f1 0 -2f8 0
t3 -2t2 2t1 2t0 -2t9 0 0 -f3 -2f2 2f1 -2f0 -2f9 0 0
t4 0 0 2t9 2t0 2t6 -2t5 -f4 0 0 2f9 -2f0 2f6 -2f5
t5 0 2t8 0 -2t6 2t0 2t4 -f5 0 2f8 0 -2f6 -2f0 2f4
t6 2t7 0 0 2t5 -2t4 2t0 -f6 2f7 0 0 2f5 -2f4 -2f0
'''


def table1(f, t):
    from .algebra import scale
    import re
    rows = []
    for row in TABLE1.strip().splitlines():
        values = []
        for token in row.split():
            if token == '0':
                values.append({})
                continue
            coefficient, family, index = re.fullmatch(r'(-?\d*)([ft])(\d)', token).groups()
            coefficient = -1 if coefficient == '-' else int(coefficient or 1)
            values.append(scale((f if family == 'f' else t)[int(index)], coefficient))
        assert len(values) == 14
        rows.append(values)
    assert len(rows) == 14
    return rows


def quaternion_table(mu1, mu2):
    # Tables 2 and 3, in the orders specified on p.18.
    return [[(1,0),(1,1),(1,2),(1,3)],
            [(1,1),(-mu1,0),(1,3),(-mu1,2)],
            [(1,2),(-1,3),(-mu2,0),(mu2,1)],
            [(1,3),(mu1,2),(-mu2,1),(-mu1*mu2,0)]]


def octonion_table(p, q, r):
    # Table 4, order 1,a,b,ab,tilde(1),tilde(a),tilde(b),tilde(ab).
    return [
        [(1,i) for i in range(8)],
        [(1,1),(-p,0),(1,3),(-p,2),(1,5),(-p,4),(-1,7),(p,6)],
        [(1,2),(-1,3),(-q,0),(q,1),(1,6),(1,7),(-q,4),(-q,5)],
        [(1,3),(p,2),(-q,1),(-p*q,0),(1,7),(-p,6),(q,5),(-p*q,4)],
        [(1,4),(-1,5),(-1,6),(-1,7),(-r,0),(r,1),(r,2),(r,3)],
        [(1,5),(p,4),(-1,7),(p,6),(-r,1),(-p*r,0),(-r,3),(p*r,2)],
        [(1,6),(1,7),(q,4),(-q,5),(-r,2),(r,3),(-q*r,0),(-q*r,1)],
        [(1,7),(-p,6),(q,5),(p*q,4),(-r,3),(-p*r,2),(q*r,1),(-p*q*r,0)],
    ]
