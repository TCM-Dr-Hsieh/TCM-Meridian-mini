"""Install-time pieces: project-relative model paths, configure.py, download_models.py helpers."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import mini.config as config_module
import mini.voice.asr as asr_module
from mini.config import DEFAULT_ALIGNER_DIR, DEFAULT_ASR_DIR, ROOT, Settings, resolve_path
from mini.voice.asr import ASRError, validate_model_dir, worker_python

sys.path.insert(0, str(ROOT / 'tools'))
import configure                    # noqa: E402
import download_models              # noqa: E402
import modelcheck                   # noqa: E402


def make_model(folder: Path, *, aligner: bool = False) -> Path:
    folder.mkdir(parents=True)
    config = {'model_type': 'qwen3_asr'}
    if aligner:
        config['timestamp_token_id'] = 1
    (folder / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    for name in ('preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (folder / name).write_text('{}', encoding='utf-8')
    (folder / 'model.safetensors').write_bytes(b'weights')
    return folder


# --- project-relative paths ------------------------------------------------------------------
def test_defaults_are_project_relative_so_the_folder_can_be_copied_anywhere():
    s = Settings()
    assert s.asr.asr_model_dir == DEFAULT_ASR_DIR and s.asr.aligner_model_dir == DEFAULT_ALIGNER_DIR
    assert not Path(DEFAULT_ASR_DIR).is_absolute() and DEFAULT_ASR_DIR.startswith('models')
    assert resolve_path(DEFAULT_ASR_DIR) == ROOT / DEFAULT_ASR_DIR


def test_relative_paths_resolve_against_the_project_root_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, 'ROOT', tmp_path)
    monkeypatch.chdir(Path(__file__).parent)                         # a different cwd must not matter
    assert resolve_path('models/x') == tmp_path / 'models' / 'x'
    absolute = tmp_path / 'elsewhere'
    assert resolve_path(str(absolute)) == absolute
    assert resolve_path(' "models/x" ') == tmp_path / 'models' / 'x'  # stray quotes/spaces from pasted paths
    s = Settings()
    s.visits_dir = 'visits'
    assert s.visits_path() == tmp_path / 'visits'


def test_model_validation_and_worker_python_use_the_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, 'ROOT', tmp_path)
    monkeypatch.setattr(asr_module, 'ROOT', tmp_path)
    make_model(tmp_path / 'models' / 'asr')
    make_model(tmp_path / 'models' / 'aligner', aligner=True)
    assert validate_model_dir('models/asr') == (tmp_path / 'models' / 'asr').resolve()
    assert validate_model_dir('models/aligner', aligner=True)
    with pytest.raises(ASRError, match='ForcedAligner'):
        validate_model_dir('models/aligner')                          # wrong model kind is reported clearly
    with pytest.raises(ASRError, match='不存在'):
        validate_model_dir('models/missing')
    s = Settings()
    assert worker_python(s.asr) == tmp_path / '.venv-asr' / 'Scripts' / 'python.exe'
    s.asr.python_path = r'tools\py\python.exe'
    assert worker_python(s.asr) == tmp_path / 'tools' / 'py' / 'python.exe'


# --- configure.py ------------------------------------------------------------------------------
def test_configure_applies_llm_settings_to_every_interface_and_clears_the_asr_python():
    s = Settings()
    s.asr.python_path = r'D:\somewhere\python.exe'
    changes = configure.apply(s, llm_url='http://10.0.0.5:8080/v1', llm_model='m1', llm_key='k', asr_python='')
    assert all(e.api_url == 'http://10.0.0.5:8080/v1' and e.model_name == 'm1' and e.api_key == 'k'
               for e in s.agents.values())
    assert s.asr.python_path == '' and len(changes) == 4
    assert configure.apply(Settings()) == []                           # no arguments: nothing changes
    s.validate()


def test_configure_keeps_per_agent_settings_it_was_not_asked_to_change():
    s = Settings()
    s.agents['professor_a'].temperature = 0.9
    configure.apply(s, llm_model='other')
    assert s.agents['professor_a'].temperature == 0.9 and s.agents['professor_a'].model_name == 'other'


# --- download_models.py --------------------------------------------------------------------------
def test_models_are_pinned_to_exact_commits_and_the_right_repos():
    asr, aligner = download_models.MODELS['asr'], download_models.MODELS['aligner']
    assert asr.repo == 'Qwen/Qwen3-ASR-1.7B' and aligner.repo == 'Qwen/Qwen3-ForcedAligner-0.6B'
    assert not asr.aligner and aligner.aligner
    for model in (asr, aligner):
        assert re.fullmatch(r'[0-9a-f]{40}', model.revision), 'revisions must be full commit SHAs'
    assert asr.folder in DEFAULT_ASR_DIR and aligner.folder in DEFAULT_ALIGNER_DIR   # defaults line up with the downloads


def manifest_for(folder: Path, repo: str) -> dict:
    """A manifest describing exactly the files currently in `folder` (SHA-256 for the weights, git blob id otherwise,
    like the real one)."""
    files = {}
    for path in sorted(folder.iterdir()):
        weights = path.suffix == '.safetensors'
        files[path.name] = {'size': path.stat().st_size,
                            'sha256': modelcheck.sha256_of(path) if weights else None,
                            'git_sha1': None if weights else modelcheck.git_blob_sha1(path)}
    return {'models': {repo: {'revision': 'x' * 40, 'files': files}}}


def test_is_valid_distinguishes_complete_incomplete_and_wrong_kind(tmp_path, monkeypatch):
    asr, aligner = download_models.MODELS['asr'], download_models.MODELS['aligner']
    good = make_model(tmp_path / 'good')
    monkeypatch.setattr(modelcheck, 'load_manifest', lambda path=modelcheck.MANIFEST_PATH: manifest_for(good, asr.repo))
    assert download_models.is_valid(good, asr) == (True, '')
    ok, why = download_models.is_valid(good, aligner)
    assert not ok and 'ForcedAligner' in why
    (good / 'model.safetensors').write_bytes(b'')                      # truncated download
    assert not download_models.is_valid(good, asr)[0]
    assert not download_models.is_valid(tmp_path / 'nothing', asr)[0]


# --- version verification against tools/model_manifest.json -------------------------------------------
def test_the_shipped_manifest_covers_both_pinned_models_and_their_weights():
    manifest = modelcheck.load_manifest()
    for model in download_models.MODELS.values():
        spec = manifest['models'][model.repo]
        assert spec['revision'] == model.revision                      # manifest and pinned commit cannot drift apart
        assert any(name.endswith('.safetensors') and info['sha256'] for name, info in spec['files'].items())
        assert all(info['size'] > 0 for info in spec['files'].values())
        for name, info in spec['files'].items():                       # a deep check must cover EVERY file
            assert bool(info['sha256']) != bool(info['git_sha1']), f'{name} needs exactly one content hash'
            assert re.fullmatch(r'[0-9a-f]{64}', info['sha256']) if info['sha256'] else \
                re.fullmatch(r'[0-9a-f]{40}', info['git_sha1'])
    assert set(modelcheck.REPOS.values()) == set(manifest['models'])


def test_git_blob_sha1_matches_what_git_hash_object_prints(tmp_path):
    empty, hello = tmp_path / 'empty', tmp_path / 'hello'
    empty.write_bytes(b'')
    hello.write_bytes(b'hello\n')
    assert modelcheck.git_blob_sha1(empty) == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    assert modelcheck.git_blob_sha1(hello) == 'ce013625030ba8dba906f756967f9e9ca394464a'
    assert modelcheck.git_blob_sha1(hello, chunk=2) == modelcheck.git_blob_sha1(hello)      # chunking changes nothing


def test_deep_check_catches_a_small_file_changed_without_changing_its_size(tmp_path):
    folder = make_model(tmp_path / 'm')
    manifest = manifest_for(folder, 'r')
    original = (folder / 'tokenizer_config.json').read_bytes()
    (folder / 'tokenizer_config.json').write_bytes(b'x' * len(original))        # same size, different content
    assert modelcheck.problems(folder, 'r', manifest=manifest) == []            # the quick size check cannot see it
    found = modelcheck.problems(folder, 'r', deep=True, manifest=manifest)
    assert len(found) == 1 and 'tokenizer_config.json' in found[0] and 'git' in found[0]
    (folder / 'tokenizer_config.json').write_bytes(original)
    assert modelcheck.problems(folder, 'r', deep=True, manifest=manifest) == []   # restored -> verified again


def test_modelcheck_reports_missing_resized_and_corrupted_files(tmp_path):
    folder = make_model(tmp_path / 'm')
    manifest = manifest_for(folder, 'r')
    assert modelcheck.problems(folder, 'r', manifest=manifest) == []
    (folder / 'extra.txt').write_text('ignored', encoding='utf-8')     # unknown extra files are fine
    assert modelcheck.problems(folder, 'r', deep=True, manifest=manifest) == []

    (folder / 'model.safetensors').write_bytes(b'WEIGHTS')             # same size, different bytes
    assert modelcheck.problems(folder, 'r', manifest=manifest) == []   # a size check cannot see it ...
    assert 'SHA-256' in modelcheck.problems(folder, 'r', deep=True, manifest=manifest)[0]   # ... the deep one does

    (folder / 'model.safetensors').write_bytes(b'short')
    assert '大小不符' in modelcheck.problems(folder, 'r', manifest=manifest)[0]
    (folder / 'config.json').unlink()
    found = modelcheck.problems(folder, 'r', manifest=manifest)
    assert any('缺少 config.json' in item for item in found) and len(found) == 2


def test_modelcheck_treats_an_unknown_repo_as_a_problem_not_a_crash(tmp_path):
    assert 'KeyError' in modelcheck.problems(tmp_path, 'nobody/nothing', manifest={'models': {}})[0]
    assert modelcheck.summarize(['a', 'b', 'c', 'd', 'e'], limit=2) == 'a；b…等共 5 項'


def run_downloader(monkeypatch, config_path, *argv):
    monkeypatch.setattr(download_models, 'CONFIG_PATH', config_path)
    monkeypatch.setattr(sys, 'argv', ['download_models.py', *argv])
    return download_models.main()


@pytest.fixture
def reused_models(tmp_path, monkeypatch):
    """Both models present in a user-chosen folder, named in config.json, and described by a matching manifest."""
    asr = make_model(tmp_path / 'mine' / 'asr')
    aligner = make_model(tmp_path / 'mine' / 'aligner', aligner=True)
    repos = download_models.MODELS
    manifest = {'models': {**manifest_for(asr, repos['asr'].repo)['models'],
                           **manifest_for(aligner, repos['aligner'].repo)['models']}}
    monkeypatch.setattr(modelcheck, 'load_manifest', lambda path=modelcheck.MANIFEST_PATH: manifest)
    settings = Settings()
    settings.asr.asr_model_dir, settings.asr.aligner_model_dir = str(asr), str(aligner)
    config_path = tmp_path / 'config.json'
    settings.save(config_path)
    return config_path, asr, aligner


def test_check_passes_for_reused_models_that_match_the_pinned_version(reused_models, monkeypatch, capsys):
    config_path, _, _ = reused_models
    assert run_downloader(monkeypatch, config_path, '--check') == 0
    assert '與固定版本相符' in capsys.readouterr().out


def test_a_reused_model_of_another_version_is_kept_but_reported(reused_models, monkeypatch, capsys):
    config_path, asr, _ = reused_models
    (asr / 'config.json').write_text('{"model_type": "qwen3_asr", "other": "version"}', encoding='utf-8')
    assert run_downloader(monkeypatch, config_path, '--check') == 1                 # --check does not accept drift
    out = capsys.readouterr().out
    assert '[警告]' in out and 'config.json 大小不符' in out
    assert run_downloader(monkeypatch, config_path, '--reuse-config') == 0           # a normal run keeps the user's model
    assert 'download' not in capsys.readouterr().out.lower()


def test_deep_verify_catches_same_size_corruption_in_a_reused_model(reused_models, monkeypatch, capsys):
    config_path, asr, _ = reused_models
    assert run_downloader(monkeypatch, config_path, '--check', '--verify') == 0
    (asr / 'model.safetensors').write_bytes(b'WEIGHTS')
    assert run_downloader(monkeypatch, config_path, '--check') == 0                  # sizes alone look fine
    capsys.readouterr()
    assert run_downloader(monkeypatch, config_path, '--check', '--verify') == 1
    assert 'SHA-256 不符' in capsys.readouterr().out


def test_a_managed_folder_that_differs_is_scheduled_for_download(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / 'config.json'
    Settings().save(config_path)
    base = tmp_path / 'managed'
    make_model(base / download_models.MODELS['asr'].folder)
    make_model(base / download_models.MODELS['aligner'].folder, aligner=True)
    # The real manifest does not describe these stand-in files, so both count as "not the pinned version".
    assert run_downloader(monkeypatch, config_path, '--dir', str(base), '--check') == 1
    out = capsys.readouterr().out
    assert out.count('[待下載]') == 2 and '與固定版本不符' in out


def test_display_path_is_relative_inside_the_project_and_absolute_outside():
    inside = ROOT / 'models' / 'Qwen3-ASR-1.7B'
    assert download_models.display_path(inside) == str(Path('models') / 'Qwen3-ASR-1.7B')
    outside = Path(ROOT.anchor) / 'somewhere-else' / 'models'
    assert download_models.display_path(outside) == str(outside)


# --- setup.ps1 (cannot be run here; guard the properties that broke before) -----------------------------
def test_setup_script_keeps_its_utf8_bom_for_windows_powershell_5_1():
    assert (ROOT / 'setup.ps1').read_bytes().startswith(b'\xef\xbb\xbf'), \
        'without a BOM PowerShell 5.1 reads the Chinese text as ANSI'


def test_setup_functions_return_one_value_even_when_a_native_command_prints():
    """First install on a computer without Python 3.12: Install-Python312 ran winget, whose output rode along with the
    returned path, so $python was an array and step 3 died with "無法辨識 'The `msstore` source requires ...'"; running
    the script a second time worked because Python was found without winget. Ensure-Venv had the same latent flaw
    (anything `python -m venv` printed would have become part of the returned venv path)."""
    shell = shutil.which('powershell') or shutil.which('pwsh')
    if not shell:
        pytest.skip('找不到 PowerShell，略過 setup.ps1 的回傳值測試')
    result = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                             str(ROOT / 'tests' / 'setup_leak_probe.ps1'), '-Setup', str(ROOT / 'setup.ps1')],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
    rows = {}
    for line in result.stdout.splitlines():
        if line.startswith('PROBE|'):
            _, name, count, kind, value = (line.split('|', 4) + [''])[:5]
            rows[name] = (int(count), kind, value)
    assert set(rows) == {'install-python', 'install-python-already-installed', 'ensure-venv'}, \
        result.stdout + result.stderr
    for name in ('install-python', 'install-python-already-installed'):
        count, kind, value = rows[name]
        assert (count, kind) == (1, 'String'), f'{name} returned {count} items ({kind}): {value}'
        assert value == r'C:\fake\Python312\python.exe'
    count, kind, value = rows['ensure-venv']
    assert (count, kind) == (1, 'String') and value.endswith(r'.venv\Scripts\python.exe'), value


def test_setup_script_refuses_a_python_that_is_not_one_working_path():
    text = (ROOT / 'setup.ps1').read_text(encoding='utf-8-sig')
    assert '$python -isnot [string] -or -not (Test-Python312 $python)' in text      # a clear message, not 無法辨識 ...
    assert text.index('$python -isnot [string]') < text.index('Write-Ok "Python 3.12：$python"')


def test_setup_script_prefers_an_explicit_models_dir_over_the_configured_folders():
    text = (ROOT / 'setup.ps1').read_text(encoding='utf-8-sig')
    assert "if ($ModelsDir) { $dlArgs += @('--dir', $ModelsDir) } else { $dlArgs += '--reuse-config' }" in text
    assert text.count("'--reuse-config'") == 1                        # never passed unconditionally
