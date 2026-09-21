import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'mechanism_internal_ablation/src'),str(ROOT/'src')]
from run_mechanism_internal_ablation import request_semantics,restore_frozen_packet_text


class RequestIdentityTests(unittest.TestCase):
    def wire(self,text,image_hash='same'):
        return {'model':'qwen','messages':[{'role':'user','content':[
            {'type':'text','text':text},
            {'type':'image_url','image_url':{'url':'[image bytes omitted]','data_url_sha256':image_hash}}]}]}

    def test_json_key_order_only_can_reuse_but_exact_packet_restored(self):
        old=self.wire('{"a":1, "b":2}')
        new=self.wire('{"b": 2,"a": 1}')
        self.assertNotEqual(old,new)
        self.assertEqual(request_semantics(old),request_semantics(new))
        self.assertEqual(restore_frozen_packet_text(new,old),old)

    def test_changed_candidate_or_image_is_not_compatible(self):
        old=self.wire('{"candidate":1}')
        self.assertNotEqual(request_semantics(old),request_semantics(self.wire('{"candidate":2}')))
        self.assertNotEqual(request_semantics(old),request_semantics(self.wire('{"candidate":1}','different')))


if __name__=='__main__':unittest.main()
