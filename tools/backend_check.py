import json,time,platform,subprocess,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def test(device):
    start=time.perf_counter();x=torch.randn(2,3,32,32,device=device);m=torch.nn.Conv2d(3,8,3,padding=1).to(device);o=torch.optim.AdamW(m.parameters(),lr=.001)
    for _ in range(8):o.zero_grad();loss=m(x).square().mean();loss.backward();o.step()
    if device=="mps":torch.mps.synchronize()
    return {"verified":True,"forward_backward_steps":8,"seconds":time.perf_counter()-start,"loss":float(loss.detach().cpu())}
report={"os":platform.system(),"os_version":platform.mac_ver()[0],"architecture":platform.machine(),"python":platform.python_version(),"cpu_count":os.cpu_count(),"ram_bytes":int(subprocess.check_output(["sysctl","-n","hw.memsize"])) ,"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"mps_built":torch.backends.mps.is_built(),"mps_available":torch.backends.mps.is_available(),"cpu":test("cpu")}
if report["mps_available"]:
    try:report["mps"]=test("mps")
    except Exception as e:report["mps"]={"verified":False,"error":repr(e)}
(ROOT/"reports/backend-check.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
