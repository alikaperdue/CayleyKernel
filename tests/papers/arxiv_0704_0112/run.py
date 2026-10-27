"""Run this paper's CPU suite; CUDA requires explicit user opt-in.

Writes exact observations, certificates, complete selected ETs, and SVG table
renderings. Default operation does not probe a CUDA device or call multiply().
"""
import argparse
import csv
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

import CayleyKernel as CK
from . import test_sections
from .emanation import labels,table,recipe_rules


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.passed=[]

    def addSuccess(self,test):
        super().addSuccess(test)
        self.passed.append(test.id())


def exact_json(value):
    if isinstance(value,Fraction):return str(value)
    raise TypeError(type(value).__name__)


def table_svg(panels,title,path):
    """Dependency-free vector rendering of exact discrete tables, not a fit."""
    size=240;gap=30;left=25;top=90
    width=left*2+len(panels)*(size+gap)-gap;height=top+size+70
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="25" y="28" font-family="sans-serif" font-size="18">{title}</text>',
         '<text x="25" y="50" font-family="sans-serif" font-size="12">Purple: marked / same slopes. Blue: unmarked / opposite slopes. White: blank.</text>']
    for panel,(label,matrix) in enumerate(panels):
        x0=left+panel*(size+gap);d=len(matrix);step=size/d
        out.append(f'<text x="{x0}" y="78" font-family="sans-serif" font-size="13">{label}</text>')
        for r,row in enumerate(matrix):
            start=0
            while start<d:
                kind=0 if row[start]==0 else 1 if row[start]>0 else -1
                end=start+1
                while end<d and (0 if row[end]==0 else 1 if row[end]>0 else -1)==kind:end+=1
                if kind:
                    colour='#297eaa' if kind>0 else '#663795'
                    out.append(f'<rect x="{x0+start*step:.5f}" y="{top+r*step:.5f}" width="{(end-start)*step:.5f}" height="{step:.5f}" fill="{colour}"/>')
                start=end
        out.append(f'<rect x="{x0}" y="{top}" width="{size}" height="{size}" fill="none" stroke="#444"/>')
        count=sum(bool(v) for row in matrix for v in row)
        out.append(f'<text x="{x0}" y="{top+size+23}" font-family="sans-serif" font-size="12">{d} by {d}; {count} filled directed cells</text>')
    out.append('<text x="25" y="'+str(height-12)+'" font-family="sans-serif" font-size="11">Rows/columns use paired XOR-S order; numeric labels and exact signed entries are in tables.json.</text></svg>')
    path.write_text('\n'.join(out)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,help='Hash the local PDF without changing it')
    parser.add_argument('--output',type=Path,help='Write report, exact data and table renderings here')
    parser.add_argument('--cuda',action='store_true',help='User opt-in: actually execute multiply() tests on a CUDA machine')
    args=parser.parse_args()
    test_sections.OBSERVATIONS.clear();test_sections.CERTIFICATES.clear()
    loader=unittest.defaultTestLoader
    suite=unittest.TestSuite([loader.loadTestsFromTestCase(test_sections.PaperVerification)])
    if args.cuda:
        suite.addTests(loader.loadTestsFromTestCase(test_sections.GPUVerification))
    stream=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
    root=Path(__file__).resolve().parents[3]
    files=sorted(Path(__file__).parent.glob('*.py'))+[root/'CayleyKernel/__init__.py', root/'tests/papers/type_replay.py']
    metadata={
        'paper':'0704.0112v3','title':'Placeholder Substructures III: A Bit-String-Driven Recipe Theory for Infinite-Dimensional Zero-Divisor Spaces',
        'paper_sha256':hashlib.sha256(args.paper.read_bytes()).hexdigest() if args.paper else None,
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'kernel_loaded':CK.loaded(),'kernel_abi':CK.abi_version(),'cuda_requested':args.cuda,
        'gpu_tests_not_selected':0 if args.cuda else 3,
        'cuda_device_probe_performed':args.cuda and CK.cp is not None and CK.loaded(),
        'elapsed_seconds':time.perf_counter()-start,'tests_run':result.testsRun,'passed':result.passed,
        'skipped':[{'test':t.id(),'reason':reason} for t,reason in result.skipped],
        'failures':[{'test':t.id(),'traceback':trace} for t,trace in result.failures],
        'errors':[{'test':t.id(),'traceback':trace} for t,trace in result.errors],
        'successful':result.wasSuccessful(),'seed':test_sections.SEED,'observations':test_sections.OBSERVATIONS,
        'scope':'Finite exact basis/ET/recipe tests; complete cells through N8, with selected samples up to N203. Neither infinite Skies, Class4 complexity, scale-free laws nor applications are proved. Printed preparation and proposed correction are separate.'}
    sys.stdout.write(stream.getvalue())
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'results.json').write_text(json.dumps(metadata,indent=2,default=exact_json)+'\n')
        (args.output/'test-output.txt').write_text(stream.getvalue())
        if result.wasSuccessful():
            (args.output/'certificates.json').write_text(json.dumps({
                'paper':'0704.0112v3','paper_sha256':metadata['paper_sha256'],'source_sha256':metadata['source_sha256'],
                'index_convention':'Mathematical zero-based indices; basis inputs/outputs are translated signed one-based labels.',
                'cases':test_sections.CERTIFICATES},indent=2,default=exact_json)+'\n')
            tables=[{'S':S,'N':N,'labels':labels(S,N),'signed_entries':table(S,N)}
                    for S,N in ((15,5),(15,6),(15,7),(25,6),(48,7),(57,7))]
            (args.output/'tables.json').write_text(json.dumps({
                'paper_sha256':metadata['paper_sha256'],'source_sha256':metadata['source_sha256'],
                'entry_convention':'0 blank; -P same-slope DMZ (marked); +P opposite-slope DMZ (unmarked). P=R XOR C.',
                'tables':tables},indent=2)+'\n')
            with (args.output/'counts.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=['N','S','G','filled_directed_cells','edges','edge_count_divided_by_12','divisible_by_24'])
                writer.writeheader()
                writer.writerows(test_sections.OBSERVATIONS['whorfian_counts'])
                writer.writerows(test_sections.OBSERVATIONS['chingon_spectrography'])
            table_svg([(f'S=15, N={N}',table(15,N)) for N in (5,6,7)],'Recomputed balloon ride: finite stages',args.output/'balloon-ride.svg')
            table_svg([(f'S={S}, N={N}',table(S,N)) for S,N in ((25,6),(48,7),(57,7))],
                      'Recomputed hide/fill tables',args.output/'hide-fill.svg')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
