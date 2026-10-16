"""V2 immutable source indices and explicit sensor/target decode boundaries."""
from __future__ import annotations
import copy,csv,datetime,functools,hashlib,json,shutil
from pathlib import Path
import numpy as np
from .experiments import root_path,json_write,sha,NYU40


def remap_nyu_labels(raw,mapping):
    raw=np.asarray(raw);mapping=np.asarray(mapping).reshape(-1)
    if not np.issubdtype(raw.dtype,np.integer) or raw.min()<0 or raw.max()>len(mapping):raise ValueError("Raw NYU label outside mapping")
    if len(mapping)!=894 or np.any((mapping<0)|(mapping>40)):raise ValueError("Invalid NYU894to40 mapping")
    ids=np.zeros(raw.shape,np.uint8);valid=raw>0;ids[valid]=mapping[raw[valid]-1]
    return np.where(ids>0,ids-1,255).astype(np.uint8)


def decode_nyu_rgb(array):
    array=np.asarray(array)
    if array.ndim!=3 or array.shape[0]!=3 or array.dtype!=np.uint8:raise ValueError("Expected MATLAB reversed RGB[C,W,H]uint8")
    return array.transpose(2,1,0).copy()


def snapshot_nyu_names(f,mapping,source):
    refs=np.array(f["names"]).reshape(-1);names=["".join(chr(int(c)) for c in np.array(f[ref]).reshape(-1)) for ref in refs]
    if len(names)!=len(mapping):raise ValueError("NYU raw label name/mapping length mismatch")
    json_write(Path(source)/"raw_class_names.json",{"source":"original officialNYUv2MAT namescellarray","raw_ids":list(range(1,len(names)+1)),"raw_names":names,"mapClass40_source_ids":list(map(int,mapping)),"NYU40_names":NYU40})


def read_index_v2(dataset,split=None,root=None):
    root=Path(root) if root is not None else root_path();key={"nyuv2":"nyu","mot":"mot17"}.get(dataset,dataset)
    path=root/"data/v2"/key/"index.jsonl";rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    groups={s:{r["scene_id"] for r in rows if r["split"]==s} for s in ["train","val","test"]}
    if any(groups[a]&groups[b] for a,b in [("train","val"),("train","test"),("val","test")]):raise ValueError("V2 scene leakage")
    return rows if split is None else [r for r in rows if r["split"]==split]


@functools.lru_cache(maxsize=64)
def _target_object(path,mtime_ns,size):
    return json.loads(Path(path).read_text())


def targets_for_row(root,row):
    if row["dataset"]=="nyu":raise ValueError("NYU semantic targets are mask_path; no detection annotations")
    path=Path(root)/row["targets_path"];stat=path.stat();obj=_target_object(str(path),stat.st_mtime_ns,stat.st_size)
    return copy.deepcopy(obj["frames"].get(str(row["frame_index"]),[]))


def _write_index(root,key,rows,report):
    out=Path(root)/"data/v2"/key;out.mkdir(parents=True,exist_ok=True);index=out/"index.jsonl"
    with index.open("w") as f:
        for row in rows:f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    report.update(index_sha256=sha(index),counts={s:sum(r["split"]==s for r in rows) for s in ["train","val","test","outside_holdout"]},
                  groups={s:sorted({r["scene_id"] for r in rows if r["split"]==s}) for s in ["train","val","test","outside_holdout"]},model_test_evaluation_performed=False)
    json_write(Path(root)/f"reports/v2/data/{key}-validation.json",report);return index


def duplicate_audit_v2(root,rows):
    """Exact SHA + v1 pHash63/Hamming<=4 and gray MAE<=8, image-only.

    Five disjoint hash bands provide a complete candidate index for distance<=4:
    at least one band is equal (pigeonhole principle). Entire lower-priority
    train/val clips are quarantined, without consulting labels/predictions.
    """
    import cv2
    cv2.setNumThreads(1);root=Path(root);fingerprints=[];buckets={};exact={};pairs=set();widths=[13,13,13,12,12];shifts=np.cumsum([0]+widths[:-1]).tolist()
    for i,row in enumerate(rows):
        path=root/row["rgb_path"];gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError("Duplicate audit RGB unreadable")
        small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32);dct=cv2.dct(small)[:8,:8].reshape(-1)[1:];bits=dct>np.median(dct);phash=sum(int(v)<<k for k,v in enumerate(bits));digest=sha(path);image=cv2.resize(gray,(64,36),interpolation=cv2.INTER_AREA).astype(np.float32)
        neighbors=set(exact.get(digest,[]))
        for band,(shift,width) in enumerate(zip(shifts,widths)):neighbors.update(buckets.get((band,(phash>>shift)&((1<<width)-1)),[]))
        for j in neighbors:
            old,oh,oi,od=fingerprints[j]
            if old["split"]!=row["split"] and ((phash^oh).bit_count()<=4 or digest==od):pairs.add((j,i))
        fingerprints.append((row,phash,image,digest));exact.setdefault(digest,[]).append(i)
        for band,(shift,width) in enumerate(zip(shifts,widths)):buckets.setdefault((band,(phash>>shift)&((1<<width)-1)),[]).append(i)
    candidates=[];quarantine=set();priority={"train":0,"val":1,"test":2}
    for i,j in sorted(pairs):
        a,ah,ai,ad=fingerprints[i];b,bh,bi,bd=fingerprints[j];mae=float(np.abs(ai-bi).mean())
        if ad!=bd and mae>8:continue
        lower=a if priority[a["split"]]<priority[b["split"]] else b;quarantine.add(lower["scene_id"])
        candidates.append({"a":a["sample_id"],"b":b["sample_id"],"group_a":a["scene_id"],"group_b":b["scene_id"],"split_a":a["split"],"split_b":b["split"],"phash_hamming":(ah^bh).bit_count(),"gray_mae":mae,"exact_file_duplicate":ad==bd})
    retained=[r for r in rows if r["scene_id"] not in quarantine]
    return retained,{"uses_ground_truth":False,"uses_predictions":False,"protocol":"crosssplitexactSHAor(phash63Hamming<=4 andgrayMAE<=8);wholelowerpriorityclipquarantine","input_samples":len(rows),"retained_samples":len(retained),"candidates":candidates,"quarantined_entire_scenes":sorted(quarantine),"candidate_index_complete_for_threshold":True,"original_recording_disjointness_proven":False}


def nyu_scene_split(scene_ids,official_train,official_test):
    """Official image split, hash-selected20% train scenes for validation.

    Any scene shared with official test is quarantined from train/val. Test
    images keep their official partition; their labels do not affect selection.
    """
    train=set(map(int,official_train));test=set(map(int,official_test))
    if train&test:raise ValueError("Official image split overlap")
    train_scenes={scene_ids[i] for i in train};test_scenes={scene_ids[i] for i in test};overlap=train_scenes&test_scenes
    eligible=train_scenes-overlap;ranked=sorted(eligible,key=lambda s:hashlib.sha256(s.encode()).hexdigest());val=set(ranked[:max(1,int(np.floor(.2*len(ranked))))])
    return {i:("test" if i in test else "quarantine" if scene_ids[i] in overlap else "val" if scene_ids[i] in val else "train") for i in train|test},sorted(overlap)


def decode_nyu(root):
    import h5py,cv2
    from scipy.io import loadmat
    cv2.setNumThreads(1);root=Path(root);source=root/"data/v2/nyu/source";out=root/"data/v2/nyu"
    validation=root/"reports/v2/data/nyu-validation.json"
    if validation.exists():
        frozen=json.loads(validation.read_text())
        if frozen.get("primary_ready"):
            if sha(out/"index.jsonl")!=frozen["index_sha256"] or sha(source/"nyu_depth_v2_labeled.mat")!=frozen["source_mat_sha256"]:raise ValueError("Frozen NYU index/source fingerprint changed")
            return out/"index.jsonl"
    mapping=loadmat(source/"classMapping40.mat")["mapClass"].reshape(-1).astype(int)
    if len(mapping)!=894 or np.any((mapping<0)|(mapping>40)):raise ValueError("Unexpected NYU894to40 mapping")
    split=loadmat(source/"splits.mat");train=split["trainNdxs"].reshape(-1).astype(int)-1;test=split["testNdxs"].reshape(-1).astype(int)-1
    if len(train)!=795 or len(test)!=654:raise ValueError("Official NYU train/test counts changed")
    for folder in ["images","masks","raw_labels","targets"]:(out/folder).mkdir(parents=True,exist_ok=True)
    rows=[];quarantine=[];hist={s:np.zeros(40,np.int64) for s in ["train","val","test"]};image_hashes={};duplicates=[]
    with h5py.File(source/"nyu_depth_v2_labeled.mat","r") as f:
        if f["images"].shape!=(1449,3,640,480) or f["labels"].shape!=(1449,640,480):raise ValueError("Unexpected MATLAB reversed-axis tensor shape")
        snapshot_nyu_names(f,mapping,source)
        refs=np.array(f["scenes"]).reshape(-1)
        scenes=["".join(chr(int(c)) for c in np.array(f[ref]).reshape(-1)) for ref in refs]
        if len(scenes)!=1449 or any(not s for s in scenes):raise ValueError("Missing NYU scene identity")
        assignment,overlap=nyu_scene_split(scenes,train,test)
        for i in range(1449):
            part=assignment[i];rgb=decode_nyu_rgb(f["images"][i]);raw=np.asarray(f["labels"][i]).T.copy()
            if rgb.dtype!=np.uint8 or rgb.shape!=(480,640,3) or raw.shape!=(480,640):raise ValueError("NYU decode shape/type")
            if raw.min()<0 or raw.max()>894:raise ValueError("NYU original class IDs outside894")
            mask=remap_nyu_labels(raw,mapping)
            stem=f"{i+1:04d}";image=out/"images"/f"{stem}.png";mask_path=out/"masks"/f"{stem}.png";raw_path=out/"raw_labels"/f"{stem}.png";target=out/"targets"/f"{stem}.json"
            if not cv2.imwrite(str(image),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3]):raise RuntimeError("NYU RGB write failure")
            if not cv2.imwrite(str(mask_path),mask) or not cv2.imwrite(str(raw_path),raw.astype(np.uint16)):raise RuntimeError("NYU target write failure")
            digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in image_hashes and image_hashes[digest][1]!=part:duplicates.append({"samples":[image_hashes[digest][0],i],"partitions":[image_hashes[digest][1],part]})
            image_hashes[digest]=(i,part)
            json_write(target,{"dataset":"nyu","source_index_1based":i+1,"scene_id":scenes[i],"mask_path":str(mask_path.relative_to(root)),"raw_label_path":str(raw_path.relative_to(root)),"mapping_sha256":sha(source/"classMapping40.mat"),"mask_encoding":"NYU40source1..40→0..39,unlabeled255"})
            row={"schema_version":2,"dataset":"nyu","domain":"real indoor","sample_id":f"nyu/{stem}","split":part,"scene_id":scenes[i],"sequence_id":scenes[i],"rgb_path":str(image.relative_to(root)),"mask_path":str(mask_path.relative_to(root)),"targets_path":str(target.relative_to(root)),"raw_label_path":str(raw_path.relative_to(root)),"width":640,"height":480,"original_width":640,"original_height":480,"decode_transform":"MATLAB axes reversed; images[N,C,W,H]→RGB[H,W,C],labels[N,W,H]→[H,W]; no resize;losslessPNG","source_index_1based":i+1,"rgb_decoded_sha256":digest,"mask_sha256":sha(mask_path)}
            if part=="quarantine":quarantine.append(row)
            else:rows.append(row);hist[part]+=np.bincount(mask[mask!=255],minlength=40)
            if (i+1)%100==0:print(json.dumps({"event":"nyu_decoded","count":i+1}),flush=True)
    # Quarantine training/validation copies of exact RGB duplicates with test.
    duplicate_train_ids={sample for d in duplicates if "test" in d["partitions"] for sample,p in zip(d["samples"],d["partitions"]) if p in {"train","val"}}
    for r in list(rows):
        if r["source_index_1based"]-1 in duplicate_train_ids:r={**r,"split":"quarantine","quarantine_reason":"exact_RGB_shared_with_test"};quarantine.append(r)
    rows=[r for r in rows if r["source_index_1based"]-1 not in duplicate_train_ids]
    (out/"quarantine.jsonl").write_text("".join(json.dumps(r)+"\n" for r in quarantine))
    json_write(out/"classes.json",{"names":NYU40,"source_ids":list(range(1,41)),"train_ids":list(range(40)),"ignore":255,"mapping_sha256":sha(source/"classMapping40.mat")})
    report={"schema_version":2,"source_mat_sha256":sha(source/"nyu_depth_v2_labeled.mat"),"splits_sha256":sha(source/"splits.mat"),"mapping_sha256":sha(source/"classMapping40.mat"),"official_counts":{"train":795,"test":654},"quarantined_scene_overlap":overlap,"quarantine_count":len(quarantine),"cross_split_exact_RGB_duplicates":duplicates,"class_histograms":{"train":hist["train"].tolist(),"val":hist["val"].tolist()},"test_labels_used_for_model_selection":False,"axis_layout_verified":True,"decoded_all_1449":True,"original_dimensions":[640,480],"ontology":"NYU40", "depth_retained_in_source_MAT_only":True}
    index=_write_index(root,"nyu",rows,report);frozen=root/f"reports/v2/data/nyu-frozen-index-{report['index_sha256'][:12]}.jsonl";shutil.copyfile(index,frozen);report.update(primary_ready=True,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_index=str(frozen.relative_to(root)));json_write(validation,report);return index


def decode_nyu_preflight(root,reader,train_only=False):
    """Sixteen train/val frames only, from the same official HDF MAT source."""
    import h5py,cv2
    from scipy.io import loadmat
    cv2.setNumThreads(1);root=Path(root);out=root/"data/v2/nyu"/("preflight-train16" if train_only else "preflight");source=root/"data/v2/nyu/source";mapping=loadmat(source/"classMapping40.mat")["mapClass"].reshape(-1).astype(int);sp=loadmat(source/"splits.mat")
    train=sp["trainNdxs"].reshape(-1).astype(int)-1;test=sp["testNdxs"].reshape(-1).astype(int)-1;rows=[]
    for folder in ["images","masks","raw_labels","targets"]:(out/folder).mkdir(parents=True,exist_ok=True)
    with h5py.File(reader,"r") as f:
        snapshot_nyu_names(f,mapping,source)
        refs=np.array(f["scenes"]).reshape(-1);scenes=["".join(chr(int(c)) for c in np.array(f[ref]).reshape(-1)) for ref in refs];assignment,overlap=nyu_scene_split(scenes,train,test)
        selected=[]
        for part in (["train"] if train_only else ["train","val"]):
            candidates=[i for i,p in assignment.items() if p==part];by_scene={}
            for i in candidates:by_scene.setdefault(scenes[i],[]).append(i)
            scene_order=sorted(by_scene,key=lambda s:hashlib.sha256(s.encode()).hexdigest());selected.extend(min(by_scene[s]) for s in scene_order[:16 if train_only else 8])
        for i in selected:
            rgb=decode_nyu_rgb(f["images"][i]);raw=np.asarray(f["labels"][i]).T.copy();mask=remap_nyu_labels(raw,mapping);stem=f"{i+1:04d}";image=out/"images"/f"{stem}.png";mp=out/"masks"/f"{stem}.png";rp=out/"raw_labels"/f"{stem}.png";tp=out/"targets"/f"{stem}.json"
            cv2.imwrite(str(image),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3]);cv2.imwrite(str(mp),mask);cv2.imwrite(str(rp),raw.astype(np.uint16))
            json_write(tp,{"source_index_1based":i+1,"raw_label_path":str(rp.relative_to(root)),"mask_path":str(mp.relative_to(root)),"preflight_only":True})
            rows.append({"schema_version":2,"dataset":"nyu","domain":"real indoor","sample_id":f"nyu-preflight/{stem}","scene_id":scenes[i],"split":assignment[i],"rgb_path":str(image.relative_to(root)),"mask_path":str(mp.relative_to(root)),"raw_label_path":str(rp.relative_to(root)),"targets_path":str(tp.relative_to(root)),"source_index_1based":i+1,"width":640,"height":480,"original_width":640,"original_height":480,"preflight_only":True,"decode_transform":"officialMAT imagesN,C,W,H→RGBH,W,C;labelsN,W,H→H,W,noresize;losslessPNG"})
            print(json.dumps({"event":"nyu_preflight_frame_ready","source_index":i+1,"split":assignment[i],"scene":scenes[i]}),flush=True)
    index=out/"index.jsonl";index.write_text("".join(json.dumps(r)+"\n" for r in rows));json_write(root/"reports/v2/data"/("nyu-preflight-train16-validation.json" if train_only else "nyu-preflight-validation.json"),{"frames":len(rows),"split_counts":{s:sum(r["split"]==s for r in rows) for s in ["train","val"]},"source_url":"https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat","source_range_total_bytes":2_972_037_809,"index_sha256":sha(index),"mapping_sha256":sha(source/"classMapping40.mat"),"preflight_only":True,"test_images_or_labels_read":False,"overlap_quarantined":overlap})
    return index


def decode_jaad_v2(root,config):
    import cv2,xml.etree.ElementTree as ET
    cv2.setNumThreads(1);root=Path(root);out=root/"data/v2/jaad";rows=[];counts={};allfiles=[];out_of_frame=0
    for part,clips in config["splits"].items():
        for clip in clips:
            annotation=out/"annotations"/f"{clip}.xml";video=out/"clips"/f"{clip}.mp4";tree=ET.parse(annotation);frames={}
            cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps<=0 or n<1:raise ValueError("JAAD invalid video metadata")
            for track_number,track in enumerate(tree.getroot().findall("track")):
                if track.attrib.get("label") not in {"pedestrian","ped","people"}:continue
                for box in track.findall("box"):
                    attrs={a.attrib.get("name"):a.text for a in box.findall("attribute")};tid=attrs.get("id",track.attrib.get("id",str(track_number)))
                    b=[float(box.attrib[k]) for k in ["xtl","ytl","xbr","ybr"]]
                    if box.attrib.get("outside")=="1" or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:continue
                    raw_occ=attrs.get("occlusion",box.attrib.get("occluded","0"));occlusion={"none":0,"part":1,"full":2}.get(raw_occ)
                    if occlusion is None:occlusion=int(raw_occ)
                    crowd=track.attrib["label"]=="people"
                    clipped=[max(0,min(w,b[0])),max(0,min(h,b[1])),max(0,min(w,b[2])),max(0,min(h,b[3]))]
                    if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:out_of_frame+=1;continue
                    frames.setdefault(str(int(box.attrib["frame"])),[]).append({"track_id":tid,"bbox_xyxy_original":b,"bbox_xyxy":clipped,"category_id":1,"category_name":"person","source_label":track.attrib["label"],"occlusion":occlusion,"iscrowd":crowd,"ignore":crowd or occlusion==2,"annotation_source":"JAAD human annotation"})
            target=out/"targets"/f"{clip}.json";json_write(target,{"video_id":clip,"split":part,"fps":fps,"width":w,"height":h,"original_width":w,"original_height":h,"bbox_scale":1.,"frames":frames})
            chosen=set(np.linspace(0,n-1,min(60,n)).round().astype(int));folder=out/"frames"/clip;folder.mkdir(parents=True,exist_ok=True)
            for i in range(n):
                ok,bgr=cap.read()
                if not ok:raise ValueError("JAAD full original video decode failure")
                if i not in chosen:continue
                image=folder/f"{i:06d}.png"
                if not cv2.imwrite(str(image),bgr,[cv2.IMWRITE_PNG_COMPRESSION,3]):raise RuntimeError("JAAD v2 image write")
                rows.append({"schema_version":2,"dataset":"jaad","sample_id":f"jaad-v2/{clip}/{i:06d}","scene_id":clip,"sequence_id":clip,"split":part,"frame_index":i,"timestamp":i/fps,"fps":fps,"rgb_path":str(image.relative_to(root)),"video_path":str(video.relative_to(root)),"targets_path":str(target.relative_to(root)),"width":w,"height":h,"original_width":w,"original_height":h,"decode_transform":"original RGB dimensions, no resize; target original XYXY","bbox_scale":1.})
            cap.release();counts[clip]=len(chosen);allfiles.append({"clip":clip,"video_sha256":sha(video),"annotation_sha256":sha(annotation),"target_sha256":sha(target)})
            print(json.dumps({"event":"jaad_v2_decoded","split":part,"clip":clip,"frames":len(chosen)}),flush=True)
    retained,audit=duplicate_audit_v2(root,rows);json_write(root/"reports/v2/data/jaad-duplicate-audit.json",audit)
    (out/"quarantine.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows if r["scene_id"] in set(audit["quarantined_entire_scenes"])))
    return _write_index(root,"jaad",retained,{"schema_version":2,"selected_config":config,"clip_counts":counts,"files":allfiles,"max_frames_per_clip":60,"full_ids_preserved":True,"original_dimensions_preserved":True,"clipped_degenerate_out_of_frame_boxes_removed":out_of_frame,"duplicate_audit":audit,"limitations":["original long-recording group IDs unavailable; clip disjoint only","testclips all fresh relativev1","crowd and fullocclusion explicitignore"]})


def decode_mot17(root):
    import configparser,cv2
    root=Path(root);base=root/"data/v2/mot17/MOT17-11-SDP";config=configparser.ConfigParser();config.read(base/"seqinfo.ini");seq=config["Sequence"]
    w,h,n,fps=[int(seq[k]) for k in ["imWidth","imHeight","seqLength","frameRate"]]
    if n!=900:raise ValueError("MOT17 requested sequence length not900")
    frames={};ignored=0;positives=0
    for record in csv.reader((base/"gt/gt.txt").open()):
        frame,tid,x,y,bw,bh,confidence,source_class,visibility=map(float,record[:9]);source_class=int(source_class)
        if source_class not in {1,2,7,8,12}:continue
        if bw<=0 or bh<=0 or not np.isfinite([x,y,bw,bh,visibility]).all():raise ValueError("MOT17 GT geometry invalid")
        ignore=confidence!=1 or source_class!=1
        target={"track_id":f"MOT17-11-SDP/{int(tid)}","bbox_xyxy_original":[x,y,x+bw,y+bh],"bbox_xyxy":[x,y,x+bw,y+bh],"category_id":1,"category_name":"person","source_class":source_class,"visibility":visibility,"ignore":ignore,"iscrowd":ignore,"annotation_source":"MOT17 official GT"}
        frames.setdefault(str(int(frame)),[]).append(target);ignored+=ignore;positives+=not ignore
    target=base/"targets.json";json_write(target,{"frames":frames,"width":w,"height":h,"fps":fps,"coordinate_convention":"MOT source XYWH→XYXY un-clipped; sourcepixelorigin preserved","positive_rule":"conf1,class1 regardless visibility; classes2/7/8/12 andconf!=1 ignored"})
    rows=[];image_hash=hashlib.sha256()
    for i in range(1,n+1):
        image=base/"img1"/f"{i:06d}.jpg";bgr=cv2.imread(str(image))
        if bgr is None or bgr.shape[:2]!=(h,w):raise ValueError("MOT17 selected RGB decode/dimensions failure")
        digest=sha(image);image_hash.update(digest.encode())
        rows.append({"schema_version":2,"dataset":"mot17","sample_id":f"mot17/MOT17-11-SDP/{i:06d}","scene_id":"MOT17-11-SDP","sequence_id":"MOT17-11-SDP","split":"outside_holdout","frame_index":i,"timestamp":(i-1)/fps,"fps":fps,"width":w,"height":h,"original_width":w,"original_height":h,"rgb_path":str(image.relative_to(root)),"targets_path":str(target.relative_to(root)),"rgb_sha256":digest,"source_partition":"MOT17 officialtrain; outsideholdoutforJAADmodel","ground_truth_access":"evaluatoronly; holdoutpredictionscloseduntilselectionlock"})
    return _write_index(root,"mot17",rows,{"schema_version":2,"sequence":"MOT17-11-SDP","dimensions":[w,h],"fps":fps,"frames":n,"image_sha256_chain":image_hash.hexdigest(),"target_sha256":sha(target),"gt_sha256":sha(base/"gt/gt.txt"),"positive_boxes":positives,"ignored_boxes":ignored,"outside_holdout":True,"predictions_evaluated":False})


def finalize_jaad_v2(root):
    """Freeze already decoded RGB, target validity and cross-split copy audit."""
    root=Path(root);index=root/"data/v2/jaad/index.jsonl";rows=[json.loads(s) for s in index.read_text().splitlines()];report_path=root/"reports/v2/data/jaad-validation.json";report=json.loads(report_path.read_text());original_sha=sha(index)
    snapshot=root/f"reports/v2/data/jaad-preaudit-index-{original_sha[:12]}.jsonl"
    if not snapshot.exists():snapshot.write_bytes(index.read_bytes())
    removed=0
    for relative in sorted({r["targets_path"] for r in rows}):
        path=root/relative;target=json.loads(path.read_text())
        for frame,boxes in target["frames"].items():
            valid=[b for b in boxes if np.isfinite(b["bbox_xyxy"]).all() and b["bbox_xyxy"][2]>b["bbox_xyxy"][0] and b["bbox_xyxy"][3]>b["bbox_xyxy"][1]]
            removed+=len(boxes)-len(valid);target["frames"][frame]=valid
        json_write(path,target)
    for item in report.get("files",[]):item["target_sha256"]=sha(root/f"data/v2/jaad/targets/{item['clip']}.json")
    retained,audit=duplicate_audit_v2(root,rows);json_write(root/"reports/v2/data/jaad-duplicate-audit.json",audit)
    quarantine=set(audit["quarantined_entire_scenes"]);(root/"data/v2/jaad/quarantine.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows if r["scene_id"] in quarantine))
    report.update(duplicate_audit=audit,clipped_degenerate_out_of_frame_boxes_removed=removed,preaudit_index_sha256=original_sha,primary_ready=True)
    final=_write_index(root,"jaad",retained,report);frozen=root/f"reports/v2/data/jaad-frozen-index-{sha(final)[:12]}.jsonl";frozen.write_bytes(final.read_bytes());return final
