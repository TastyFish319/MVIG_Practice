"""制作可离线浏览的统计目录与轻量ZIP；权重绝不打包。"""
from pathlib import Path
from html import escape
import csv
import json
import shutil
import zipfile
from acttrain.data.core import read_json, write_json

STYLE = "body{font:16px system-ui;background:#f4f7fb;color:#19273a;max-width:1250px;margin:36px auto;padding:20px;line-height:1.7}header,section{background:white;border-radius:16px;padding:24px;margin:20px 0;box-shadow:0 5px 24px #2233440a}h1{font-size:34px}h2{color:#235878}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #e0e7ee}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#126b9a}img{max-width:100%}.tag{color:#237864}"

def sanitized(value):
    if isinstance(value,dict):
        return {k:sanitized(v) for k,v in value.items() if k not in ('path','dataset','output','train_argv','evaluate_argv','log','supervisor_pid','checkpoint')}
    if isinstance(value,list):
        return [sanitized(v) for v in value]
    return value

def build_report(run, evaluation=None, light=True, output=None):
    run=Path(run).resolve()
    evaluation=Path(evaluation).resolve() if evaluation else run/'evaluation'
    if read_json(evaluation/'progress.json').get('status')!='passed':
        raise ValueError('仅能为完整成功的评测制作最终统计包')
    summary=read_json(evaluation/'summary.json')
    if summary.get('status')!='passed' or not summary.get('checkpoints'):
        raise ValueError('评测汇总不完整')
    target=Path(output).resolve() if output else run/'statistics'
    stage=target.with_name(target.name+'.partial')
    archive=run/(run.name+'_'+target.name+'.zip')
    if target.exists() or stage.exists() or archive.exists():
        raise FileExistsError('统计目录、partial或ZIP已存在，请选择新输出目录')
    stage.mkdir(parents=True)
    # Include every locally linked report asset; .workers, caches and weights are excluded.
    for source, destination in ((evaluation,stage/'evaluation'), (run/'metrics',stage/'metrics')):
        if not source.is_dir():
            continue
        for item in source.rglob('*'):
            relative=item.relative_to(source)
            if not item.is_file() or item.is_symlink() or any(p.startswith('.') for p in relative.parts):
                continue
            allowed={'.html','.png','.json','.csv'} | (set() if light else {'.npz','.svg'})
            if item.suffix.lower() not in allowed:
                continue
            dest=destination/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if item.suffix=='.json':
                write_json(dest,sanitized(read_json(item)))
            else:
                shutil.copy2(item,dest)
    settings=read_json(run/'config.json') if (run/'config.json').is_file() else {}
    for name in ('config.json','best_validation.json','checkpoint_retention.json'):
        if (run/name).is_file():
            write_json(stage/name,sanitized(read_json(run/name)))
    ranking=[]
    for index,item in enumerate(summary['checkpoints'],1):
        ranking.append({'rank':index,'checkpoint':Path(item['checkpoint']).parent.name,
                        'score':item['ranking_score'],'first4_joint_mae_deg':item['errors']['first4']['joint_mae_deg'],
                        'full_joint_mae_deg':item['errors']['full']['joint_mae_deg'],
                        'gripper_mae':item['errors']['full']['gripper_mae']})
    with (stage/'ranking.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(ranking[0]));writer.writeheader();writer.writerows(ranking)
    rows=''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in r.values())+'</tr>' for r in ranking)
    head=''.join(f'<th>{escape(k)}</th>' for k in ranking[0])
    selection=read_json(evaluation/'selection.json') if (evaluation/'selection.json').exists() else {}
    html=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{escape(run.name)} · ACT统计</title><style>{STYLE}</style><header><span class="tag">ACT / 离线实验报告</span><h1>{escape(run.name)}</h1><p>从录制观测预测未来动作；以下分数越低越好。离线误差不是机器人任务成功率。</p></header>'
    html+='<section><h2>训练曲线</h2><a href="metrics/curves.html">打开完整曲线与指标说明</a><img src="metrics/training_curves.png" alt="训练及验证loss"><p>比较train-prior和val-prior；posterior训练loss使用不同条件。</p></section>'
    html+=f'<section><h2>Checkpoint排名</h2><table><tr>{head}</tr>{rows}</table><p><a href="evaluation/index.html">逐checkpoint → 逐episode → 每关节曲线与CSV</a> · <a href="ranking.csv">排名CSV</a></p></section>'
    html+='<section><h2>数据与评测样本</h2><pre>'+escape(json.dumps(sanitized(selection),ensure_ascii=False,indent=2))+'</pre><h2>本次有效配置</h2><pre>'+escape(json.dumps(sanitized(settings),ensure_ascii=False,indent=2))+'</pre></section>'
    html+='<section><h2>阅读顺序</h2><ol><li>查看验证曲线是否稳定，再看最佳权重是否早于训练结束。</li><li>对比前4步与完整32步误差，优先检查部署实际执行的前4步。</li><li>展开逐关节曲线，看leader目标、follower实测与模型预测的时间对齐。</li></ol><p>关节曲线单位为度，CSV/NPZ为rad；夹爪为归一化开度。夹爪过滤不参与排名。</p></section></html>'
    (stage/'report.html').write_text(html,encoding='utf-8')
    from html.parser import HTMLParser
    from urllib.parse import unquote,urlsplit
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src') and value and not urlsplit(value).scheme and not value.startswith('#'):
                    resolved=(self.file.parent/unquote(value.split('#')[0])).resolve()
                    if not resolved.is_relative_to(stage.resolve()) or not resolved.is_file():
                        raise ValueError(f'离线报告链接缺失或越界: {self.file.name}: {value}')
    for page in stage.rglob('*.html'):
        parser=Links();parser.file=page;parser.feed(page.read_text(encoding='utf-8'))
    write_json(stage/'bundle.json',{'schema_version':1,'light':light,'weights_included':False,
               'checkpoint_count':len(ranking),'entry':'report.html'})
    stage.rename(target)
    temporary=archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for item in sorted(target.rglob('*')):
            if item.is_file():bundle.write(item,item.relative_to(target).as_posix())
    temporary.rename(archive)
    return archive
