"""Build a local 1× source-clock replay viewer from verified chapter evidence."""
from __future__ import annotations
import argparse,datetime,json,math,os
from pathlib import Path
import cv2
from indoor_perception_lab.experiments import root_path,sha,json_write
ROOT=root_path()

HTML='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Indoor Perception · quality v2 replay</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#101820;color:#edf3f6}*{box-sizing:border-box}body{margin:0}header{padding:22px 30px;border-bottom:1px solid #30434c;display:flex;align-items:center;justify-content:space-between}h1{font-size:21px;margin:0 0 5px}p{margin:0;color:#a8bac4}.badge{border:1px solid #a88c48;color:#efcc79;padding:7px 12px;border-radius:4px}main{padding:22px 30px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:22px}.stage{background:#070d12;min-height:440px;display:grid;place-items:center;border:1px solid #30434c}.stage img{display:block;max-height:64vh;width:100%;object-fit:contain}button{background:#213440;border:1px solid #49616e;border-radius:4px;color:#eaf5f9;padding:9px 14px;cursor:pointer}button:hover,button[aria-current=true]{background:#395566}.controls{display:flex;gap:10px;align-items:center;margin:14px 0}input{width:100%;accent-color:#6ad3cd}.time{font-variant-numeric:tabular-nums;white-space:nowrap}.chapter{width:100%;text-align:left;display:block;margin-bottom:8px;font-size:13px;line-height:1.4}h2{font-size:15px;margin:20px 0 12px}.fact{display:flex;justify-content:space-between;border-bottom:1px solid #293e49;padding:9px 0;gap:12px}.fact span{color:#a8bac4}.fact strong{font-weight:500;text-align:right;overflow-wrap:anywhere}.note{font-size:12px;line-height:1.65;color:#a8bac4;margin-top:16px}a{color:#80d5db}.source{font:12px ui-monospace,monospace;overflow-wrap:anywhere;margin:12px 0}footer{padding:0 30px 22px;font-size:12px;color:#98afbd}.error{color:#ff998d}@media(max-width:900px){main{grid-template-columns:1fr;padding:15px}header{padding:16px}.stage{min-height:220px}.stage img{max-height:56vh}footer{padding:15px}}
</style></head><body><header><div><h1>Indoor Perception · quality v2</h1><p>Recorded robot sensors and controlled simulation · local evidence replay</p></div><div class="badge">PARTIAL · physical / embedded tests open</div></header><main><section><div class="stage"><img id="panel" alt="Recorded perception output"></div><div class="controls"><button id="play">Play</button><button id="reset">Restart</button><span id="time" class="time"></span><span class="time">1× source time</span></div><input id="seek" aria-label="Replay time" type="range" min="0" step="0.05"><div id="source" class="source"></div><p id="error" class="error" role="status"></p></section><aside><div id="chapters"></div><h2>Displayed frame</h2><div class="fact"><span>Chapter time</span><strong id="chaptertime"></strong></div><div class="fact"><span>Sensor depth</span><strong id="depth"></strong></div><div class="fact"><span>Pose source</span><strong id="pose"></strong></div><div class="fact"><span>Observed tracks</span><strong id="tracks"></strong></div><div class="fact"><span>Replay RPC</span><strong id="latency"></strong></div><div class="fact"><span>Sim command</span><strong id="command"></strong></div><p class="note" id="role"></p><p class="note">Each chapter keeps its original sensor clock. Chapters are separate recordings or scenarios. RPC values include IPC and are separate from formal repeated benchmark statistics. Semantic colors and detector scores are uncalibrated outputs.</p><p class="note"><a id="result" target="_blank">Frame result JSON</a> · <a id="chapterevidence" target="_blank">Chapter evidence</a> · <a href="manifest.json" target="_blank">Replay manifest</a></p></aside></main><footer>Independent candidate portfolio. The real robot chapter is recorded data; simulation commands actuate a virtual robot. Physical robot and embedded-device performance remain unverified. Dataset-derived media stays local; source terms are in the data card.</footer><script id="evidence" type="application/json">__MANIFEST__</script><script>
const data=JSON.parse(document.getElementById('evidence').textContent);const $=id=>document.getElementById(id);let time=0,playing=false,lastClock=null,lastKey=null;const total=data.source_duration_s;$('seek').max=total;
const buttons=data.chapters.map((c,i)=>{const b=document.createElement('button');b.className='chapter';b.textContent=`${i+1}. ${c.label} · ${c.source_duration_s.toFixed(1)} s`;b.onclick=()=>{time=c.timeline_start_s;lastKey=null;show();};$('chapters').appendChild(b);return b;});
function locate(){const c=data.chapters.find(c=>time<c.timeline_end_s)||data.chapters.at(-1);const local=Math.min(c.source_duration_s,Math.max(0,time-c.timeline_start_s));let f=c.frames[0];for(const next of c.frames){if(next.time_s>local)break;f=next;}return {c,f,local};}
function show(){const {c,f,local}=locate();const key=c.chapter_id+'/'+f.sample_id;if(key!==lastKey){lastKey=key;$('panel').src=f.image;$('source').textContent=f.sample_id+' · original timestamp '+f.timestamp_ns+' ns';$('depth').textContent=f.depth_available?'available':'missing';$('pose').textContent=f.pose_source||'unavailable';$('tracks').textContent=f.observed_tracks??'unavailable';$('latency').textContent=Number.isFinite(f.rpc_ms)?f.rpc_ms.toFixed(1)+' ms':'unavailable';$('command').textContent=f.command?`${f.command.linear_mps.toFixed(2)} m/s · ${f.command.reason}`:'recorded sensors only';$('role').textContent=c.model_role;$('result').href=f.result_ref;$('chapterevidence').href=c.evidence_ref;buttons.forEach((b,i)=>b.setAttribute('aria-current',String(data.chapters[i]===c)));}$('chaptertime').textContent=local.toFixed(1)+' s';$('time').textContent=time.toFixed(1)+' / '+total.toFixed(1)+' s';$('seek').value=time;}
$('panel').onerror=()=>{$('error').textContent='A verified frame could not be loaded. Open this viewer with the documented local HTTP server.';playing=false;$('play').textContent='Play';};$('panel').onload=()=>{$('error').textContent='';};
$('play').onclick=()=>{if(time>=total)time=0;playing=!playing;lastClock=null;$('play').textContent=playing?'Pause':'Play';};$('reset').onclick=()=>{time=0;lastKey=null;show();};$('seek').oninput=e=>{time=+e.target.value;lastClock=null;show();};
function tick(now){if(playing&&lastClock!==null){time=Math.min(total,time+(now-lastClock)/1000);if(time>=total){playing=false;$('play').textContent='Play';}}lastClock=now;show();requestAnimationFrame(tick);}show();requestAnimationFrame(tick);
</script></body></html>'''

def build(chapter_paths,output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'demo/v2'):raise ValueError('V2 replay output must remain under demo/v2')
    output.mkdir(parents=True,exist_ok=True);chapters=[];start=0.;decoded=0;chapter_ids=set()
    def resolve(reference,parent):
        path=Path(reference);path=(path if path.is_absolute() else parent/path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Missing/out-of-project media reference')
        return path
    def url(path):return os.path.relpath(path,output)
    for requested in chapter_paths:
        path=resolve(requested,ROOT);c=json.loads(path.read_text())
        if c.get('status')!='verified' or not c.get('all_source_frames_processed') or c.get('playback_speed')!=1.:raise ValueError('Executed source-clock chapter evidence required')
        if c['chapter_id'] in chapter_ids:raise ValueError('Duplicate replay chapter ID')
        chapter_ids.add(c['chapter_id'])
        if c.get('source_original_manifest'):
            original=resolve(c['source_original_manifest'],path.parent)
            if sha(original)!=c['source_original_manifest_sha256']:raise ValueError('Original chapter proof changed')
        duration=float(c['source_duration_s'])
        if not math.isfinite(duration) or duration<=0 or not c['frames']:raise ValueError('Invalid actual chapter duration')
        frames=[];previous=-1.
        for f in c['frames']:
            if not math.isfinite(f['time_s']) or f['time_s']<0 or f['time_s']<=previous or f['time_s']>duration or not frames and f['time_s']!=0:raise ValueError('Chapter frames must begin at zero and use increasing in-range source times')
            previous=f['time_s'];image=resolve(f['image'],path.parent);result=resolve(f['result_ref'],path.parent)
            image_hash=f.get('image_sha256',c.get('verification',{}).get('panel_sha256',{}).get(f['image']))
            result_hash=f.get('result_sha256',c.get('verification',{}).get('result_sha256',{}).get(f['result_ref']))
            if image_hash!=sha(image) or result_hash!=sha(result):raise ValueError('Missing/changed declared replay frame evidence hash')
            if cv2.imread(str(image)) is None:raise ValueError('Undecodable replay evidence PNG')
            decoded+=1;reply=json.loads(result.read_text());r=reply.get('result',reply);tracks=r.get('tracks')
            # Sensor nanoseconds commonly exceed JavaScript's exact integer range.
            frames.append({**f,'timestamp_ns':str(f['timestamp_ns']),'image':url(image),'image_sha256':sha(image),'result_ref':url(result),'result_sha256':sha(result),'observed_tracks':sum(bool(t.get('observed',True)) for t in tracks) if tracks is not None else None,'rpc_ms':reply.get('actual_rpc_elapsed_ms',reply.get('adapter_ms')),'command':reply.get('control',{}).get('command')})
        chapter={**{k:c[k] for k in ['chapter_id','label','source_duration_s','model_role','processed_sensor_frames','clock_policy']},'frames':frames,'timeline_start_s':start,'timeline_end_s':start+duration,'evidence_ref':url(path),'evidence_sha256':sha(path)}
        chapters.append(chapter);start+=duration
    if not 60<=start<=90:raise ValueError('Actual source-clock demo must cover60–90seconds')
    manifest={'schema_version':1,'protocol_id':'quality-v2','overall_status':'PARTIAL','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_duration_s':start,'playback_speed':1.,'chapters':chapters,'decoded_verified_panel_frames':decoded,'media_role':'actual inference outputs on recorded/simulated sensors; no generated illustrations','physical_robot_validation':False,'embedded_target_validation':False}
    json_write(output/'manifest.json',manifest);embedded=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    (output/'index.html').write_text(HTML.replace('__MANIFEST__',embedded))
    report={'status':'verified','output':str(output.relative_to(ROOT)),'source_duration_s':start,'chapters':len(chapters),'verified_images':decoded,'manifest_sha256':sha(output/'manifest.json'),'viewer_sha256':sha(output/'index.html'),'visual_browser_QA':'pending'}
    json_write(ROOT/'reports/v2/demo-build.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--chapters',nargs='+',default=['reports/v2/robotics/media/openloris/chapter-manifest.json','reports/v2/robotics/media/mujoco-depth-loss/chapter-manifest.json','reports/v2/robotics/media/mujoco-occlusion/chapter-manifest.json','reports/v2/forecast/media/jaad-validation/viewer-chapter-manifest.json']);p.add_argument('--output',default='demo/v2');a=p.parse_args();print(json.dumps(build(a.chapters,ROOT/a.output)))
