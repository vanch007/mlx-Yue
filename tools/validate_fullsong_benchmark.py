"""Verify full-song provenance, measured durations, assets and termination."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from yue2.storage import verify_result
from build_unified_showcase import prepare_case, status_label
from run_fullsong_benchmark import latest_directory

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / 'reports/official-fullsong'
    manifest = json.loads((report / 'cases.json').read_text())
    checks = []
    for case in manifest['cases']:
        check = {'id': case['id'], 'status': 'fail'}
        try:
            row = prepare_case(case)
            m = row['measurement']
            if m is None or not status_label(m).startswith('pass'):
                raise ValueError(status_label(m))
            path = latest_directory(case['id']) / 'song'
            result = verify_result(path, m['identity'])
            config = json.loads((path / 'config.json').read_text())
            assert config['generation']['semantic']['max_tokens'] == 9000
            assert config['generation']['ode_steps'] == 32
            info = sf.info(path / 'audio.flac')
            assert info.samplerate == 48000 and info.channels == 2
            assert abs(info.duration - m['audio_seconds']) < 1 / 48000
            assert abs(info.duration - result['audio_seconds']) < 1 / 48000
            assert abs(m['rtf'] - m['e2e_seconds'] / info.duration) < 1e-9
            frames, power, peak = 0, 0., 0.
            for block in sf.blocks(path / 'audio.flac', blocksize=48000, dtype='float64'):
                assert np.isfinite(block).all()
                frames += len(block)
                power += float(np.square(block).sum())
                peak = max(peak, float(np.abs(block).max()))
            assert frames == info.frames and power > 0
            for key in ('official_player', 'local_player'):
                mp3 = sf.info(ROOT / 'docs' / row[key])
                expected = info.duration if key == 'local_player' else case['official_seconds']
                assert abs(mp3.duration - expected) < .1, 'Listening copy is shortened or stale'
            check.update(status='pass', audio_seconds=info.duration, decoded_frames=frames,
                         rms=float(np.sqrt(power / (frames * 2))), peak=peak,
                         duration_ratio=m['duration_ratio'], musical_quality='pending')
        except Exception as error:
            check['error'] = f'{type(error).__name__}: {error}'
        checks.append(check)
    result = {'status': 'pass' if all(c['status'] == 'pass' for c in checks) else 'fail',
              'scope': 'Artifact integrity, full decode, input matching, native budget, RTF, no truncation. Not perceptual quality.',
              'cases': checks}
    (report / 'audio-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
