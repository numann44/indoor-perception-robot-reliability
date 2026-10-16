import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
from indoor_perception_lab.data_v2 import remap_nyu_labels,decode_nyu_rgb,nyu_scene_split,read_index_v2,targets_for_row,duplicate_audit_v2

ROOT=Path(__file__).resolve().parents[2]


class NyuDataTests(unittest.TestCase):
    def test_mapping_one_based_40_classes_and_ignore(self):
        m=np.ones(894,int);m[-1]=40;m[10]=0
        np.testing.assert_array_equal(remap_nyu_labels(np.array([[0,1,894,11]],np.uint16),m),[[255,0,39,255]])
        with self.assertRaises(ValueError):remap_nyu_labels(np.array([[895]]),m)
    def test_axis_orientation_and_channel_order(self):
        original=np.arange(2*4*3,dtype=np.uint8).reshape(2,4,3);matlab=original.transpose(2,1,0)
        np.testing.assert_array_equal(decode_nyu_rgb(matlab),original)
        self.assertEqual(decode_nyu_rgb(matlab).shape,(2,4,3))
    def test_official_split_overlap_scene_quarantined(self):
        scenes=["shared","trainA","trainB","trainC","trainD","trainE","shared","testA"]
        split,overlap=nyu_scene_split(scenes,range(6),[6,7])
        self.assertEqual(overlap,["shared"]);self.assertEqual(split[0],"quarantine");self.assertEqual(split[6],"test")
        val={scenes[i] for i,p in split.items() if p=="val"};expected=min(scenes[1:6],key=lambda s:hashlib.sha256(s.encode()).hexdigest());self.assertEqual(val,{expected})
    def test_image_index_overlap_rejected(self):
        with self.assertRaises(ValueError):nyu_scene_split(["a","b"],[0,1],[1])
    def test_targets_format_fullids_and_crowd(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);(root/"target.json").write_text(json.dumps({"frames":{"17":[{"track_id":"0_27_119b","bbox_xyxy":[1,2,3,4],"ignore":False},{"track_id":"0_27_119","bbox_xyxy":[9,8,7,6],"ignore":True,"iscrowd":True}]}}))
            targets=targets_for_row(root,{"dataset":"jaad","targets_path":"target.json","frame_index":17});self.assertEqual([t["track_id"] for t in targets],["0_27_119b","0_27_119"])
            targets[0]["bbox_xyxy"][0]=999;self.assertEqual(targets_for_row(root,{"dataset":"jaad","targets_path":"target.json","frame_index":17})[0]["bbox_xyxy"][0],1)
    def test_index_scene_leakage_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);out=root/"data/v2/jaad";out.mkdir(parents=True);(out/"index.jsonl").write_text(json.dumps({"scene_id":"same","split":"train"})+"\n"+json.dumps({"scene_id":"same","split":"test"})+"\n")
            with self.assertRaisesRegex(ValueError,"scene leakage"):read_index_v2("jaad",root=root)
    def test_duplicate_audit_quarantines_entire_lower_priority_clip_without_targets(self):
        import cv2
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);rng=np.random.default_rng(5);image=rng.integers(0,255,(64,96,3),dtype=np.uint8);different=rng.integers(0,255,image.shape,dtype=np.uint8)
            for name,pixels in [("a.png",image),("b.png",image),("c.png",different)]:cv2.imwrite(str(root/name),pixels)
            rows=[{"sample_id":"train-copy","scene_id":"train-clip","split":"train","rgb_path":"a.png"},{"sample_id":"train-other","scene_id":"train-clip","split":"train","rgb_path":"c.png"},{"sample_id":"test-copy","scene_id":"test-clip","split":"test","rgb_path":"b.png"}]
            kept,audit=duplicate_audit_v2(root,rows);self.assertEqual([r["sample_id"] for r in kept],["test-copy"]);self.assertEqual(audit["quarantined_entire_scenes"],["train-clip"]);self.assertFalse(audit["uses_ground_truth"]);self.assertFalse(audit["uses_predictions"])


if __name__=="__main__":unittest.main()
