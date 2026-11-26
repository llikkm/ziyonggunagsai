"""Read-only postprocessing: audit saved results, export empirical P draws and report."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from protocol import check_lock
from solids import SEEDS


def report(root):
    check_lock()
    lines=['# 冻结模块 09：独立泛化测试报告','','固定版本 0b82f3b；几何模型为统计分组单位，同件的六次扫描为重复测量。']
    for split,title in [('development','开发验证集'),('holdout','最终保留测试集')]:
        folder=root/split
        rows=list(csv.DictReader((folder/'results.csv').open(encoding='utf-8-sig')))
        assert len(rows)==len({(r['model'],r['mode'],r['pose']) for r in rows})==54
        summary=json.loads((folder/'summary.json').read_text())
        lines.extend(['',f'## {title}（种子 {SEEDS[split]}）','',f'[逐次结果]({split}/results.csv) · [清单]({split}/models/manifest.json) · [完整精度汇总]({split}/summary.json)','',
                      '| 扫描模式 | 类别 | 有效/总数 | 绝对相对误差中位数 % | P95 % | 最大 % |',
                      '|---|---|---:|---:|---:|---:|'])
        draws={}
        for mode,stat in summary.items():
            selected=[r for r in rows if r['mode']==mode or mode=='pooled']
            rng=np.random.default_rng(SEEDS[split]+191)
            samples=np.zeros(10000)
            for category,weight in zip('ABC',[.2,.3,.5]):
                table=stat['classes'][category]
                quantiles=table['abs_pct']
                values=' | '.join(f'{v:.3f}' for v in quantiles.values()) if quantiles else '缺测 | 缺测 | 缺测'
                lines.append(f"| {mode} | {category} | {table['n']-table['failures']}/{table['n']} | {values} |")
                errors=[float(r['abs_pct']) if not r['failure'] else np.nan for r in selected if r['category']==category]
                samples+=weight*rng.choice(errors,10000)/100
            draws[mode]=samples
            assert (~np.isfinite(samples)).sum()==stat['invalid_draws']
        np.savez_compressed(folder/'P_draws.npz',**draws)
        lines.extend(['','| 随机 ABC 组合 | 有效/抽取 | P 中位数 | P95 | 最大值 |','|---|---:|---:|---:|---:|'])
        for mode,stat in summary.items():
            quantiles=stat['unrounded_P']
            values=' | '.join(f'{v:.8f}' for v in quantiles.values()) if quantiles else '缺测 | 缺测 | 缺测'
            lines.append(f"| {mode} | {10000-stat['invalid_draws']}/10000 | {values} |")
        good=[r for r in rows if not r['failure']]
        bad=[r for r in rows if r['failure']]
        lines.extend(['',f'失败 {len(bad)}/54。各阶段异常保存在逐次 CSV；P 分布在 [{split}/P_draws.npz]({split}/P_draws.npz)，失败组合保留 NaN。','',
                      '|类别|最差成功样本及摆放|有符号误差|直接/估计覆盖率|主要归因|','|---|---|---:|---:|---|'])
        for category in 'ABC':
            subset=[r for r in good if r['category']==category]
            if not subset:
                continue
            row=max(subset,key=lambda r:float(r['abs_pct']))
            lines.append(f"| {category} | {row['kind']} · p{row['pose']} · {row['mode']} | {float(row['signed_pct']):+.3f}% | {float(row['direct_pct']):.2f}% / {float(row['estimated_pct']):.2f}% | {row['dominant']} |")
        lines.extend(['',f"主导项计数：{dict(Counter(r['dominant'] for r in rows))}。"])
        for reason,count in Counter(r['failure'] for r in bad).items():
            lines.append(f'- {count} 次：{reason}')
        if good:
            run=np.array([float(r['runtime_s']) for r in good])
            lines.append(f"成功扫描耗时中位数 {np.median(run):.2f} s；P95 {np.quantile(run,.95):.2f} s。")
        for row in rows:
            case=folder/f"{row['model']}_p{row['pose']}_{row['mode']}"
            assert len(list((case/'measurements').glob('*.png')))==63
            assert (case/'measurements/config.json').is_file()
            if not row['failure']: assert (case/'grid.npz').is_file() and (case/'reconstruction/cloud.ply').is_file()
    lines.extend(['','## 诊断结论','',
        '开发集 54 次中，边界项主导 23 次、扫描范围项 19 次。保留集相应为 37 次和 12 次。球冠在 p2 理想扫描低估 6.493%，邻近位姿估算的范围损失约 3.98%；即使 p0 居中，扰动扫描仍低估约 5.28%。仅扩大行程难以解释全部球冠误差。',
        'B 保留集最坏的槽孔组合、C 最坏的弯脊都出现在 p2（旋转 53°、平移 −9/+7 mm），范围项主导。当前视场对居中标准件的验证不能直接推广到旋转平移后的较高物体。',
        '开发/保留集中表面高度分别主导 2/1 次。B 孔槽缺测项没有位居净体积误差首位，但在重建网格覆盖的真实凹部，局部缺测率最高分别达 6.82% 和 8.36%。各案例独立复核见 [开发机制表](development/mechanisms.csv) 和 [保留机制表](holdout/mechanisms.csv)。',
        '扫描剖面单区间假设：俯视真足迹在重建网格内未发现多区间 X 列，也没有有序剖面异常。本批测试无法量化违反该假设时的退化幅度。各类别只有三件独立保留模型，姿态重复与随机组合不增加独立形状数。保留集已用于本报告；后续改版应另留新测试集。'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    report(parser.parse_args().root)
