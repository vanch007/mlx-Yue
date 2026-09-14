# Security Trust Report

- OK: `True`
- Scanned files: `24`
- Scripts: `7`
- Internal script modules: `0`
- Secret findings: `0`
- Network-capable scripts: `0`
- Network policy covered scripts: `0`
- Network policy missing scripts: `0`
- File-write scripts: `2`
- Permission approvals: `0 / 2`
- Permission approval gaps: `2`
- CLI help smoke checked: `0`
- CLI help smoke failures: `0`
- Interactive scripts: `0`
- Package hash scope: `source-contract-without-generated-reports`
- Package hash files: `24`
- Package SHA256: `0aaf7d5bce51024f379f7a5ebe489ca8c099b7a98a93d2e70337a64988fecc64`

## Failures

- None

## Warnings

- No dependency or lock file detected
- CLI scripts without argparse/help surface: scripts/abc_tools.py, scripts/common.py, scripts/listen.py, scripts/run_yue2.py, scripts/transcribe.py
- Permission approvals missing: file_write, subprocess

## Dependency Evidence

- Files: `none`
- Pinned entries: `0`
- Unpinned entries: `0`

## Network Policy

- Policy file: `security/network_policy.json`
- Present: `False`
- Covered scripts: `0`
- Missing scripts: `none`
- Mismatches: `0`

## Permission Governance

- Policy file: `security/permission_policy.json`
- Present: `False`
- Required capabilities: `file_write, subprocess`
- Approved capabilities: `none`
- Missing approvals: `file_write, subprocess`
- Invalid approvals: `none`
- Expired approvals: `none`

## CLI Help Smoke

- Enabled: `False`
- Timeout seconds: `5.0`
- Checked scripts: `0`
- Passed scripts: `0`
- Failed scripts: `none`

## Script Surface

| Script | Interface | Declared | Argparse | Main Guard | Input | Network | File Write | Subprocess | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scripts/abc_tools.py | cli | False | False | True | False | False | False | False | Default CLI classification; add SCRIPT_INTERFACE for internal modules. |
| scripts/common.py | cli | False | False | False | False | False | False | False | Default CLI classification; add SCRIPT_INTERFACE for internal modules. |
| scripts/compare_steps.py | cli | False | True | True | False | False | True | False | Default CLI classification; add SCRIPT_INTERFACE for internal modules. |
| scripts/generate_music.py | cli | False | True | True | False | False | True | True | Default CLI classification; add SCRIPT_INTERFACE for internal modules. |
| scripts/listen.py | cli | False | False | True | False | False | False | False | Default CLI classification; add SCRIPT_INTERFACE for internal modules. |
| scripts/run_yue2.py | cli | False | False | True | False | False | False | False | Default CLI classification; add SCRIPT_INTERFACE for internal modules. |
| scripts/transcribe.py | cli | False | False | True | False | False | False | False | Default CLI classification; add SCRIPT_INTERFACE for internal modules. |
