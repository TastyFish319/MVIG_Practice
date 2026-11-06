"use strict";
let tasks=window.REVIEW_TASKS||[];
const el=id=>document.getElementById(id);
function element(tag,cls,text){const node=document.createElement(tag);node.className=cls;if(text!==undefined)node.textContent=text;return node}
function render(){
 el('totalTasks').textContent=tasks.length;
 el('totalEpisodes').textContent=tasks.reduce((n,t)=>n+t.episodes,0);
 el('totalApproved').textContent=tasks.reduce((n,t)=>n+(t.approved||0),0);
 const search=el('taskSearch').value.trim().toLowerCase();
 const visible=tasks.filter(t=>(t.title+' '+t.source).toLowerCase().includes(search));
 el('taskCount').textContent=tasks.length?visible.length+' 个任务':'';el('tasks').replaceChildren();
 for(const task of visible){
  const a=element('a','task-card');a.href='/tasks/'+encodeURIComponent(task.id)+'/index.html';
  const top=element('div','task-card-top');top.append(element('span','task-glyph','▤'),element('span','pill',task.episodes+' 条演示'));
  const meta=element('div','task-card-meta');meta.append(element('span','','检查通过 '+(task.summary.pass||0)),element('span','','需复查 '+(task.summary.warning||0)),element('span','','技术失败 '+(task.summary.fail||0)));
  const bottom=element('div','task-card-bottom');bottom.append(element('span','',`已审核 ${task.reviewed||0} / ${task.episodes}`),element('strong','','继续审核 →'));
  a.append(top,element('h3','',task.title),element('p','task-source',task.source),meta,bottom);el('tasks').append(a);
 }
 if(!visible.length){const box=element('div','empty-state');box.append(element('h3','',tasks.length?'没有匹配的任务':'还没有生成审核任务'),element('p','',tasks.length?'换一个关键词，或清空搜索。':'双击 prepare_task.cmd 选择原始数据并生成回放，即可在这里开始。'));el('tasks').append(box)}
}
el('taskSearch').addEventListener('input',render);render();
if(location.protocol==='http:')fetch('/api/catalog').then(r=>{if(!r.ok)throw Error();return r.json()}).then(data=>{tasks=data.tasks;render()}).catch(()=>{});
