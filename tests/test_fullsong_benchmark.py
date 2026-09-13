"""Regression guards for the withdrawn short/mismatched demo report."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


def tool(name):
    path = Path(__file__).resolve().parents[1] / 'tools' / (name + '.py')
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_truncated_and_missing_results_never_pass():
    page = tool('build_unified_showcase')
    assert page.status_label(None).startswith('pending')
    assert page.status_label({'status': 'complete'}).startswith('missing evidence')
    # The original result.json called a capped run "complete" too.
    assert page.status_label({'status': 'complete', 'truncated': {
        'abc': False, 'semantic': True}}).startswith('fail')
    assert page.status_label({'status': 'complete', 'truncated': {
        'abc': False, 'semantic': False}}) == 'pass · 完成'


def test_official_text_score_and_audio_must_match(tmp_path):
    runner = tool('run_fullsong_benchmark')
    audio = tmp_path / 'reference.mp3'
    audio.write_bytes(b'reference audio fixture')
    case = {'request': {'style': 'pop', 'lyrics': 'complete lyrics', 'abc': 'X:1\nK:C\nC4|'},
            'source': {'tags': 'pop', 'lyrics': 'complete lyrics', 'abc': 'X:1\nK:C\nC4|'},
            'score_mode': 'supplied', 'official_audio_path': str(audio),
            'official_audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest()}
    runner.validate_case(case)
    for key in ('lyrics', 'style', 'abc'):
        altered = copy.deepcopy(case)
        altered['request'][key] = 'rewritten'
        with pytest.raises(ValueError, match='mismatch'):
            runner.validate_case(altered)
    capped = copy.deepcopy(case)
    capped['request']['semantic_sampling'] = {'max_tokens': 800}
    with pytest.raises(ValueError, match='budgets'):
        runner.validate_case(capped)
    audio.write_bytes(b'another song')
    with pytest.raises(ValueError, match='identity mismatch'):
        runner.validate_case(case)


def test_batch_failure_is_not_hidden_by_successful_controller():
    runner = tool('run_fullsong_benchmark')
    good = {'status': 'complete', 'truncated': {'abc': False, 'semantic': False}}
    assert runner.batch_exit_code([good]) == 0
    for bad in ({'status': 'failed'}, {'status': 'complete'},
                {'status': 'complete', 'truncated': {'abc': False, 'semantic': True}}):
        assert runner.batch_exit_code([good, bad]) == 1


def test_retry_preserves_failure_and_does_not_fall_back_to_older_audio(tmp_path):
    runner = tool('run_fullsong_benchmark')
    runner.ROOT = tmp_path
    runner.OUT = tmp_path / 'outputs'
    first = runner.attempt_directory('case')
    first.mkdir(parents=True)
    runner.write(first / 'measurement.json', {'id': 'case', 'status': 'failed', 'error': 'AC disconnected'})
    original_hash = runner.digest(first / 'measurement.json')
    second = runner.attempt_directory('case', 2)
    second.mkdir()
    assert runner.latest_directory('case') == second
    assert runner.measurement_for('case') is None
    runner.write(second / 'measurement.json', {'id': 'case', 'status': 'complete'})
    receipt = runner.measurement_for('case')
    assert receipt['prior_attempts'][0]['receipt_sha256'] == original_hash
    assert runner.digest(first / 'measurement.json') == original_hash
