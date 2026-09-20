import importlib.util
import os
import subprocess
import sys
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('manager',Path(__file__).with_name('manager.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def gguf(path):
    def s(text):
        b=text.encode();return struct.pack('<Q',len(b))+b
    fields=s('general.name')+struct.pack('<I',8)+s('Small model')
    fields+=s('tokenizer.ggml.tokens')+struct.pack('<IIQ',9,8,2)+s('one')+s('two')
    fields+=s('general.architecture')+struct.pack('<I',8)+s('qwen4exp')
    fields+=s('qwen4exp.context_length')+struct.pack('<II',4,32768)
    path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,4)+fields)

class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.patcher=patch.object(m,'DATA',self.root/'state');self.patcher.start();self.addCleanup(self.patcher.stop)
        m.atomic_json(m.DATA/'settings.json',{'roots':[str(self.root/'models')],'profiles':{}})
        (self.root/'models').mkdir()
        self.file=self.root/'models'/'model-Q8_0-00001-of-00002.gguf';gguf(self.file)
        self.second=self.file.with_name('model-Q8_0-00002-of-00002.gguf');gguf(self.second)
        # vision defaults on, and the projector is looked for beside the model. Without this the
        # suite passed only on a machine that happened to have a real one at the old default path.
        self.mmproj=self.file.with_name(m.MMPROJ_NAME);gguf(self.mmproj)
        self.model={'id':'model','path':str(self.file),'architecture':'qwen4exp','context':32768,'role':'model'}
        # a fake runtime, so the suite runs on a machine that has never built one
        rt=self.root/'bin'/'hip-rocm101';rt.mkdir(parents=True)
        for name in ('llama-server.exe','ggml-hip.dll'):(rt/name).write_bytes(b'')
        rocm=self.root/'toolchain'/'rocm';rocm.mkdir(parents=True);(rocm/'amdhip64_7.dll').write_bytes(b'')
        for attr,value in (('RUNTIME',rt/'llama-server.exe'),('ROCM_BIN',rocm)):
            pat=patch.object(m,attr,value);pat.start();self.addCleanup(pat.stop)
    def test_metadata_skips_tokenizer_and_reads_later_fields(self):
        self.assertEqual(m.metadata(self.file)['qwen4exp.context_length'],32768)
    def test_catalog_groups_shards_and_marks_missing(self):
        first=[x for x in m.catalog(True)['models'] if x['role']=='model'];self.assertEqual(len(first),1)
        self.assertEqual(first[0]['size'],self.file.stat().st_size*2)
        self.second.unlink()
        self.assertEqual(m.catalog(True)['models'][0]['missing'],[self.second.name])
    def test_model_path_cannot_escape_registered_roots(self):
        outside=self.root/'outside.gguf';gguf(outside)
        with self.assertRaises(ValueError):m.checked_file(outside)
        with self.assertRaises(ValueError):m.checked_file(self.second)
    def test_invalid_profiles_fail_before_launch(self):
        for values in [{'context':999999},{'context':True},{'ubatch':1024,'batch':512},{'kv':'q8_0'}, {'flash_attention':'auto'}, {'flash_attention':True}, {'flash_attention':['on']},{'draft_min':float('nan')},{'command':'calc.exe'}]:
            with self.subTest(values=values),self.assertRaises(ValueError):m.validate_profile(values,self.model)
    def test_argv_keeps_paths_as_single_arguments(self):
        cfg=m.validate_profile({'mtp':False},self.model)
        args=m.argv(self.model,cfg)
        self.assertEqual(args[args.index('-m')+1],str(self.file))
        self.assertIn('f16',args);self.assertNotIn('--spec-type',args)
    def test_large_batches_survive_save_and_reach_launch_without_starting_model(self):
        model=m.catalog(True)['models'][0]
        with patch.object(m,'state',return_value={}),patch.object(m.subprocess,'Popen') as popen:
            for batch,ubatch in ((4096,4096),(8192,4096),(8192,8192),(16384,16384),(32768,32768)):
                with self.subTest(batch=batch,ubatch=ubatch):
                    m.handle('save',{'id':model['id'],'profile':{'mtp':False,'batch':batch,'ubatch':ubatch}})
                    cfg=m.profile(model)
                    self.assertEqual((cfg['batch'],cfg['ubatch']),(batch,ubatch))
                    args=m.argv(model,cfg)
                    self.assertEqual(args[args.index('-b')+1],str(batch))
                    self.assertEqual(args[args.index('-ub')+1],str(ubatch))
            popen.assert_not_called()
        for values in ({'batch':32769},{'batch':32768,'ubatch':32769},
                       {'batch':4096,'ubatch':8192},{'batch':31},{'ubatch':31}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                m.validate_profile({'mtp':False,**values},model)
    def test_ngram_spec_legacy_default_and_invalid_values(self):
        model=m.catalog(True)['models'][0]
        settings=m.settings();settings['profiles'][model['id']]={'mtp':False,'context':8192}
        m.atomic_json(m.DATA/'settings.json',settings)
        before=(m.DATA/'settings.json').read_bytes()
        cfg=m.validate_profile({},model)
        self.assertIs(cfg['ngram_spec'],False)
        self.assertEqual(cfg['context'],8192)
        self.assertNotIn('--spec-type',m.argv(model,cfg))
        self.assertEqual((m.DATA/'settings.json').read_bytes(),before)
        for value in ('true',1,None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                m.validate_profile({'ngram_spec':value},model)
    def test_ngram_save_start_is_independent_of_mtp_and_embedding_placement(self):
        from types import SimpleNamespace
        model=m.catalog(True)['models'][0]
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        with patch.object(m,'ROOT',self.root),patch.object(m,'discover',return_value=[]), \
             patch.object(m,'process_identity',return_value=ident), \
             patch.object(m,'http_json',side_effect=m.urllib.error.URLError('no listener')), \
             patch('socket.socket'),patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as popen:
            for mtp,ngram in ((False,False),(False,True),(True,False),(True,True)):
                if True:
                    with self.subTest(mtp=mtp,ngram=ngram):
                        popen.reset_mock()
                        m.handle('save',{'id':model['id'],'profile':{'mtp':mtp,'ngram_spec':ngram,
                            'draft':str(self.file) if mtp else str(self.root/'missing.gguf'),
                            'draft_max':3,'draft_min':0.3}})
                        popen.assert_not_called()
                        self.assertIs(m.profile(model)['ngram_spec'],ngram)
                        result=m.handle('start',{'id':model['id']})
                        args=popen.call_args.args[0]
                        expected=','.join((['ngram-mod'] if ngram else [])+(['draft-mtp'] if mtp else []))
                        self.assertEqual(args.count('--spec-type'),int(bool(expected)))
                        if expected:self.assertEqual(args[args.index('--spec-type')+1],expected)
                        self.assertEqual('-md' in args,mtp)
                        # this runtime reads the per-layer embedding table itself, never pinned to CPU
                        self.assertNotIn('-ot',args)
                        self.assertIn('--lazy-mode',args)
                        self.assertEqual('--spec-ngram-mod-n-max' in args,ngram)
                        if ngram:
                            for option,value in (('--spec-ngram-mod-n-match','24'),('--spec-ngram-mod-n-min','4'),('--spec-ngram-mod-n-max','8')):
                                self.assertEqual(args[args.index(option)+1],value)
                        if mtp:
                            self.assertEqual(args[args.index('--spec-draft-n-max')+1],'3')
                            self.assertEqual(args[args.index('--spec-draft-p-min')+1],'0.3')
                        self.assertEqual(args[args.index('-ctk')+1],'f16')
                        self.assertEqual(args[args.index('-ctv')+1],'f16')
                        self.assertIs(result['profile']['ngram_spec'],ngram)
                        self.assertIn('n-gram draft=on' if ngram else 'n-gram draft=off',Path(result['log']).read_text())
                        m.handle('stop',{})
    def kwargs_for(self,thinking):
        import json
        cfg=m.validate_profile({'mtp':False,'thinking':thinking},self.model)
        args=m.argv(self.model,cfg)
        self.assertIn('--cache-prompt',args)
        return json.loads(args[args.index('--chat-template-kwargs')+1])
    def test_thinking_levels_map_to_the_template_s_reasoning_effort(self):
        # the chat template raises on anything outside low/medium/xhigh, so these are not free-form
        self.assertEqual(self.kwargs_for('off'),{'enable_thinking':False})
        for level,effort in (('low','low'),('medium','medium'),('high','xhigh')):
            self.assertEqual(self.kwargs_for(level),
                             {'enable_thinking':True,'reasoning_effort':effort})
    def test_thinking_still_accepts_the_boolean_it_used_to_be(self):
        self.assertEqual(self.kwargs_for(False),{'enable_thinking':False})
        self.assertEqual(self.kwargs_for(True),{'enable_thinking':True,'reasoning_effort':'xhigh'})
    def test_thinking_rejects_a_level_the_template_would_raise_on(self):
        for bad in ('false','xhigh','extra-high',3):
            with self.assertRaises(ValueError):m.validate_profile({'thinking':bad},self.model)
    def test_profile_normalises_the_old_boolean_thinking_and_drops_unknown_fields(self):
        model=m.catalog(True)['models'][0]
        settings=m.settings();settings['profiles'][model['id']]={'thinking':True,'ngram_cpu':True,'context':8192}
        m.atomic_json(m.DATA/'settings.json',settings)
        cfg=m.profile(model)
        # the page displays what profile() returns, so a stored switch must already be a level
        self.assertEqual(cfg['thinking'],'high');self.assertNotIn('ngram_cpu',cfg);self.assertEqual(cfg['context'],8192)
        # and sends it straight back: an earlier version's field must not make the whole profile unknown
        self.assertEqual(m.validate_profile({**cfg,'mtp':False},model)['context'],8192)
    def test_shared_memory_is_off_first_and_a_forced_profile_turns_it_on(self):
        with patch.object(m,'dedicated_vram_bytes',return_value=96*2**30):
            cfg=m.validate_profile({'mtp':False},self.model)
            self.assertFalse(m.unified_memory(self.model,cfg))
            self.assertEqual(m.runtime_environment(cfg,False)['GGML_HIP_ENABLE_UNIFIED_MEMORY'],'0')
            self.assertEqual(m.runtime_environment(cfg,True)['GGML_HIP_ENABLE_UNIFIED_MEMORY'],'1')
            self.assertTrue(m.unified_memory(self.model,m.validate_profile({'mtp':False,'shared_vram':True},self.model)))
    def running(self,alive,dedicated=64*2**30):
        """The launch fixture: a fake runtime, no listener on 8080, and process_identity answering
        from `alive` - a queue whose last entry repeats, so [None, ident] is one dead check and
        then a live process."""
        from types import SimpleNamespace
        from contextlib import ExitStack
        stack=ExitStack()
        for p in (patch.object(m,'ROOT',self.root),patch.object(m,'discover',return_value=[]),
                  patch.object(m,'dedicated_vram_bytes',return_value=dedicated),
                  patch.object(m,'process_identity',side_effect=lambda pid,*a,**k: alive.pop(0) if len(alive)>1 else alive[0]),
                  patch.object(m,'http_json',side_effect=m.urllib.error.URLError('no listener')),
                  patch('socket.socket')):
            stack.enter_context(p)
        popen=stack.enter_context(patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=123)))
        return stack,popen
    def test_out_of_memory_falls_back_to_shared_memory_once_and_is_remembered(self):
        model=m.catalog(True)['models'][0]
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        alive=[ident]
        stack,popen=self.running(alive)
        with stack:
            first=m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            self.assertEqual(popen.call_args.kwargs['env']['GGML_HIP_ENABLE_UNIFIED_MEMORY'],'0')
            self.assertFalse(first['unified']);self.assertIn('shared memory=off',Path(first['log']).read_text())
            # the load dies of out-of-memory in the dedicated carve; the relaunched process lives
            with open(first['log'],'a') as f:f.write('ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3488.00 MiB on device 0: cudaMalloc failed: out of memory\n')
            alive[:]=[None,ident]
            status=m.status()
            self.assertEqual(popen.call_count,2)
            self.assertEqual(popen.call_args.kwargs['env']['GGML_HIP_ENABLE_UNIFIED_MEMORY'],'1')
            self.assertEqual((status['status'],status['notice'],status['unified']),('loading','shared_vram_fallback',True))
            self.assertNotIn('failure',status)
            alive[0]=ident
            self.assertEqual(m.status()['notice'],'shared_vram_fallback')
            self.assertEqual(popen.call_count,2)
            # remembered: the next load of this model, at this carve, with this profile, starts in shared memory
            m.handle('stop',{})
            m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            self.assertEqual(popen.call_args.kwargs['env']['GGML_HIP_ENABLE_UNIFIED_MEMORY'],'1')
            m.handle('stop',{})
            # ...but a different carve, or a profile that takes different memory, is tried afresh
            with patch.object(m,'dedicated_vram_bytes',return_value=96*2**30):
                m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            self.assertEqual(popen.call_args.kwargs['env']['GGML_HIP_ENABLE_UNIFIED_MEMORY'],'0')
            m.handle('stop',{})
            m.handle('start',{'id':model['id'],'profile':{'mtp':False,'context':4096}})
            self.assertEqual(popen.call_args.kwargs['env']['GGML_HIP_ENABLE_UNIFIED_MEMORY'],'0')
            m.handle('stop',{})
    def test_a_death_in_shared_memory_is_reported_not_retried(self):
        model=m.catalog(True)['models'][0]
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        alive=[ident]
        stack,popen=self.running(alive)
        with stack:
            first=m.handle('start',{'id':model['id'],'profile':{'mtp':False,'shared_vram':True}})
            self.assertTrue(first['unified'])
            with open(first['log'],'a') as f:f.write('llama_kv_cache: failed to allocate buffer for kv cache\n')
            alive[0]=None
            status=m.status()
            self.assertEqual(popen.call_count,1)
            self.assertEqual(status['status'],'stopped');self.assertIn('failed to allocate',status['failure'])
            # a plain exit - the app closing takes the server with it - is not a failure to report
            m.handle('stop',{});alive[0]=ident
            second=m.handle('start',{'id':model['id'],'profile':{'mtp':False}})
            alive[0]=None
            self.assertNotIn('failure',m.status())
            m.handle('stop',{})
            self.assertNotIn('exited',m.read_json(m.DATA/'process.json',{}))
            self.assertNotIn('shared memory=on',Path(second['log']).read_text())
    def test_installed_layout_needs_no_sdk_directory_and_no_path_entry(self):
        # tools/make_runtime_bundle.py puts the ROCm DLLs beside llama-server
        with patch.object(m,'ROCM_BIN',self.root/'nowhere'):
            self.assertFalse(m.runtime_available())
            (m.RUNTIME.parent/'amdhip64_7.dll').write_bytes(b'')
            self.assertTrue(m.runtime_available())
            env=m.runtime_environment(m.validate_profile({'mtp':False},self.model))
            self.assertNotIn('nowhere',env['PATH'])
    def test_draft_falls_back_to_the_family_s_shared_head_beside_the_model(self):
        # an installed copy has no models/ of ours; Unsloth's mtp-*.gguf sits next to the model
        shared=self.file.with_name('mtp-'+m.MODEL_FAMILY+'-shared-Q4_K_M.gguf');gguf(shared)
        other=self.file.with_name('mtp-Other-Model-shared-Q4_K_M.gguf');gguf(other)
        m.catalog(True)
        with patch.object(m,'DEFAULT_DRAFT',self.root/'models'/'missing-head.gguf'):
            # the catalog stores resolved paths; the temp dir may be an 8.3 short name
            self.assertEqual(Path(m.family_draft()).resolve(),shared.resolve())
            model={**self.model,'path':str(self.file.with_name(m.MODEL_FAMILY+'-UD-IQ4_XS.gguf'))}
            self.assertEqual(Path(m.profile(model)['draft']).resolve(),shared.resolve())
            shared.unlink();m.catalog(True)
            with self.assertRaisesRegex(ValueError,'草稿'):m.validate_profile({'mtp':True},model)
    def test_saving_configuration_does_not_start_a_process(self):
        m.catalog(True)
        model=m.catalog()['models'][0]
        with patch.object(m,'state',return_value={}),patch.object(m.subprocess,'Popen') as popen:
            result=m.handle('save',{'id':model['id'],'profile':{'context':8192,'mtp':False}})
            self.assertEqual(result['profile']['context'],8192);popen.assert_not_called()
        self.assertEqual(m.profile(model)['context'],8192)
    def test_flash_attention_survives_save_and_reaches_launch_arguments(self):
        model=m.catalog(True)['models'][0]
        self.assertEqual(m.profile(model)['flash_attention'],'off')
        with patch.object(m,'state',return_value={}),patch.object(m.subprocess,'Popen') as popen:
            for mode in ('on','off'):
                m.handle('save',{'id':model['id'],'profile':{'mtp':False,'flash_attention':mode}})
                cfg=m.validate_profile(m.profile(model),model)
                args=m.argv(model,cfg)
                self.assertEqual(args[args.index('-fa')+1],mode)
                self.assertEqual(args[args.index('-ctk')+1],'f16')
                self.assertEqual(args[args.index('-ctv')+1],'f16')
            popen.assert_not_called()
    def test_unknown_operation_rejected(self):
        with self.assertRaises(ValueError):m.handle('execute',{'cmd':'calc.exe'})
    def test_runtime_allowlist_is_exact_and_status_uses_live_binary(self):
        runtime=m.RUNTIME
        self.assertTrue(m.managed_runtime(runtime))
        self.assertFalse(m.managed_runtime(runtime.parent/'other.exe'))
        ident={'pid':123,'exe':str(runtime.resolve()),'birth':456}
        with patch.object(m,'state',return_value={'identity':ident,'adopted':True}),patch.object(m,'http_json',return_value={'status':'ok','data':[]}):
            status=m.status()
            self.assertEqual(status['runtime'],str(runtime.resolve()))
            # a binary at the right path proves nothing about the gates it was started with
            self.assertNotIn('runtime_env',status)
        self.assertFalse(m.managed_runtime(self.root/'some-other-build'/'llama-server.exe'))
    def test_candidate_can_be_adopted_and_stopped_by_exact_identity(self):
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        entry={'ProcessId':123,'ExecutablePath':ident['exe'],'CommandLine':'llama-server.exe --host 127.0.0.1 --port 8080'}
        with patch.object(m,'discover',return_value=[entry]),patch.object(m,'process_identity',return_value=ident) as identity:
            adopted=m.state()
            self.assertTrue(adopted['adopted'])
            self.assertEqual(adopted['identity'],ident)
            self.assertNotIn('runtime_env',adopted)
            m.handle('stop',{})
            identity.assert_called_with(123,True,ident)
        self.assertNotIn('identity',m.read_json(m.DATA/'process.json',{}))

    def test_qsa_legacy_profile_defaults_off(self):
        model=m.catalog(True)['models'][0]
        settings=m.settings();settings['profiles'][model['id']]={'context':16384,'flash_attention':'on'}
        m.atomic_json(m.DATA/'settings.json',settings)
        cfg=m.validate_profile({},model)
        self.assertFalse(cfg['qsa'])
        self.assertEqual(m.selected_runtime(cfg),m.RUNTIME)
        self.assertEqual(cfg['context'],16384)
    def test_qsa_invalid_dependencies_rejected(self):
        for values in ({'qsa':'true'},{'qsa':1},{'qsa':None},
                       {'qsa':True,'flash_attention':'off'},
                       {'qsa':True,'flash_attention':'off'}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                m.validate_profile(values,self.model)
        with self.assertRaisesRegex(ValueError,'qwen4exp'):
            m.validate_profile({'qsa':True,'flash_attention':'on'},{**self.model,'architecture':'qwen35moe'})
    def test_qsa_save_start_stop_roundtrip_preserves_model_settings(self):
        from types import SimpleNamespace
        model=m.catalog(True)['models'][0]
        baseline=m.validate_profile({'context':16384,'flash_attention':'on','mtp':True,
                                     'draft':str(self.file),'draft_max':3,'draft_min':0.35},model)
        settings=m.settings();settings['profiles'][model['id']]=baseline
        m.atomic_json(m.DATA/'settings.json',settings)
        ident={'pid':123,'exe':str(m.RUNTIME.resolve()),'birth':456}
        with patch.object(m,'ROOT',self.root),patch.object(m,'discover',return_value=[]),              patch.object(m,'process_identity',return_value=ident) as identity,              patch.object(m,'http_json',side_effect=m.urllib.error.URLError('no listener')),              patch('socket.socket'),patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as popen:
            popen.reset_mock()
            m.handle('save',{'id':model['id'],'profile':{'qsa':True}})
            popen.assert_not_called()
            cfg=m.profile(model)
            self.assertEqual({**cfg,'qsa':False},baseline)
            result=m.handle('start',{'id':model['id']})
            args=popen.call_args.args[0];env=popen.call_args.kwargs['env']
            self.assertEqual(args[0],str(m.RUNTIME))
            for option,value in (('-fa','on'),('-ctk','f16'),('-ctv','f16'),
                                 ('--spec-draft-n-max','3'),('--spec-draft-p-min','0.35')):
                self.assertEqual(args[args.index(option)+1],value)
            # this runtime has no context threshold: the gates are simply on
            self.assertEqual(env['LLAMA_QSA_SPARSE'],'1')
            self.assertEqual(env['LLAMA_QSA_QUERY_STRIP'],'512')
            self.assertEqual(result['runtime_env']['LLAMA_QSA_SPARSE'],'1')
            self.assertIn('QSA=on',Path(result['log']).read_text())
            self.assertEqual(m.status()['runtime_env'],result['runtime_env'])
            m.handle('stop',{});identity.assert_called_with(123,True,ident)
            m.handle('save',{'id':model['id'],'profile':{'qsa':False}})
            self.assertEqual(m.selected_runtime(m.profile(model)),m.RUNTIME)

    def test_log_polling_preserves_partial_utf8(self):
        log=self.root/'logs'/'unicode.log';log.parent.mkdir()
        log.write_bytes(b'line\n'+'中'.encode()[:2])
        with patch.object(m,'ROOT',self.root),patch.object(m,'state',return_value={'log':str(log)}):
            first=m.logs();self.assertEqual(first['text'],'line\n')
            with log.open('ab') as f:f.write('中'.encode()[2:])
            self.assertEqual(m.logs(first['offset'])['text'],'中')
    @unittest.skipUnless(os.name=='nt','Windows process identity')
    def test_process_guard_rejects_pid_reuse_and_other_executable(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)'],creationflags=m.HIDDEN)
        try:
            ident=m.process_identity(child.pid)
            self.assertIsNotNone(ident)
            with self.assertRaises(ValueError):m.process_identity(child.pid,True,{**ident,'birth':ident['birth']+1})
            with self.assertRaises(ValueError):m.process_identity(child.pid,True,ident)
            self.assertIsNone(child.poll())
        finally:
            child.terminate();child.wait(timeout=5)

if __name__=='__main__':unittest.main()
