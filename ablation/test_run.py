import json, unittest
from unittest.mock import patch
import run

class Contracts(unittest.TestCase):
    def setUp(self):
        self.c=next(c for c in run.read(run.ROOT/'cases.json')['cases'] if c['id']=='expert-kim-in-flowery-clouds-02__m11-12')
        self.r=run.Runner.__new__(run.Runner)
        self.r.settings=run.read(run.ROOT/'settings.json')
    def test_source_quote_uses_actual_citation_not_the_reference_answer(self):
        c={**self.c,'reference_answers':['REFERENCE_MUST_NEVER_ENTER_GENERATION']}
        draft={'raw':'설명입니다. [E1]','visible':'설명입니다.','warning':None,'useful':True,'strict_ok':True,'prompt_hash':'same'}
        q=self.r.quotation(c,c['fixed_evidence'],draft)
        self.assertEqual(q['quote_ids'],[c['fixed_evidence'][0]['record']['id']])
        self.assertEqual(q['visible'],'참고 자료의 원문입니다.\n'+c['fixed_evidence'][0]['record']['answer'])
        self.assertNotIn('REFERENCE_MUST',q['raw'])
    def test_no_citation_does_not_invent_quote_selection(self):
        d={'raw':'일반 설명입니다.','visible':'일반 설명입니다.','prompt_hash':'same'}
        q=self.r.quotation(self.c,self.c['fixed_evidence'],d)
        self.assertFalse(q['quote_applied']);self.assertEqual(q['visible'],d['visible'])
    def test_guard_failure_never_rewrites_shared_draft(self):
        pool=self.c['fixed_evidence'];d={'raw':'설명입니다. [E1]','visible':'설명입니다.','warning':None,'useful':True,'strict_ok':True,'prompt_hash':'same'}
        def scoped(stage):return run.contextmanager(lambda: iter([None]))()
        self.r.scope=scoped
        self.r.guarded=lambda payload,stage:{'allow':stage=='admission','reason':'security_output_rejected'}
        self.r.compose=lambda c,p:dict(d)
        self.r.safety_only=lambda c,t:True
        self.r.calls=[]
        with patch.object(run.security,'select_question_evidence',side_effect=lambda a,b,payload,**kw:payload),patch.object(run,'append'):
            self.r.generate=lambda *a,**k:''
            out={o['variant']:o for o in self.r.one(self.c)}
        self.assertEqual(out['g0s0q0']['response']['answer'],'설명입니다.')
        self.assertEqual(out['g1s0q0']['response']['generation_mode'],'unavailable')
        self.assertEqual(out['g0s0q0']['diagnostic']['shared_draft_hash'],out['g1s0q0']['diagnostic']['shared_draft_hash'])
        self.assertIn('설명입니다.',out['warning']['response']['answer'])
        self.assertIsNotNone(out['warning']['diagnostic']['visible_warning'])
    def test_only14b_frozen(self):
        self.assertEqual(self.r.settings['model_filename'],'Qwen3-14B-Q6_K.gguf')
        self.assertEqual(len(run.read(run.ROOT/'cases.json')['cases']),233)

if __name__=='__main__':unittest.main()
