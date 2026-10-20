"""Render the unapproved B17 BRIEF for installation as campaign/BRIEF.md after review."""
import argparse
import json
from pathlib import Path

from nrp.common import CAMPAIGN, digest
from tools.retries import POLICY


def draft(root, admission=None):
    root = Path(root)
    frozen = {}
    for directory in ('nrp', 'tools', 'gen', 'cbl'):
        for p in sorted((root / directory).glob('*.py')):
            frozen['code/' + str(p.relative_to(root))] = digest(p.read_bytes())
    for rel in ('tools/admission.json', 'configs/MANIFEST.json', 'manifests/data-prep-150c.json'):
        frozen['code/' + rel] = digest((root / rel).read_bytes())
    frozen['ADMISSION.json'] = digest(Path(admission or root / 'tools/admission.json').read_bytes())
    command = ['python3', '-B', '-m', 'tools.operations']
    common = ['--campaign-dir', '{cdir}', '--job', '{job}']
    auth = {
        'campaign': CAMPAIGN, 'gpu_h_max': 400, 'reserve_gpu_h': {'reserve': 50},
        'stage_order': ['S1', 'S3', 'S4', 'reserve'], 'gpu_bound_mode': 'hard',
        'max_concurrent_gpu_jobs': 4, 'elapsed_days_max': None, 'elapsed_behavior': 'stop_new_submissions',
        'candidate_attempts_max': 0, 'repairs_total_max': 0, 'infra_relaunches_max': 0,
        'agent_invocations_max': 0, 'runtime_factor_max': 1,
        'approval_ref': 'CBL1016-B17-DRAFT: Kai terminal approval after PREFLIGHT (STUDY v0.10 section 11)',
        'namespace': 'cms-ml', 'selector': 'campaign=' + CAMPAIGN, 'notify_channels': ['file'],
        'dispatch_actions': [], 'runner': 'codex', 'implement_runner': 'codex', 'review_runner': 'codex',
        'codex_model': None, 'runner_command': None, 'interpret_command': None, 'workspace': 'code',
        'repair_paths': [], 'candidate_paths': [], 'candidate_protected': ['**'], 'interpret_paths': [],
        'scientific_settings': {'b17_retry_policy': dict(POLICY)}, 'frozen_files': frozen,
        'repair_test_command': command + ['repair-check'] + common,
        'prepare_command': command + ['relaunch'] + common + ['--attempt', '{attempt}', '--cls', '{cls}'],
        'workspace_command': command + ['workspace'] + common,
        'evaluate_command': command + ['evaluate'] + common,
        'candidate_check_command': None, 'candidate_prepare_command': None,
    }
    ops = {'lease_s': 3600, 'infra_relaunches_per_key': 0, 'repair_attempts_per_class': 0,
           'agent_invocations_warn': None, 'elapsed_days_warn': None, 'deadline_seconds_per_epoch': 3,
           'candidate_runtime_factor': 1, 'runtime_factor_notify': 1, 'score_deadline_s': 600,
           'extra_excluded_nodes': [], 'kubectl': '/home/kaimoe/.local/bin/kubectl', 'codex_bin': 'codex'}
    prose = '''# Campaign brief: chang-bitlinear-20261016 (B17 draft)

Unapproved. No APPROVAL.json is created. Install the reviewed text as the campaign's BRIEF.md;
Kai runs lab-root `python3 tools/harness.py approve <campaign>` at a terminal after PREFLIGHT.
No launch, smoke, benchmark or draw precedes that approval (STUDY v0.10 §11, NRP).

The question and frozen scientific rules are in review/snapshots/v0.10/STUDY.md. Code arrives
only as a complete, immutable git bundle on kai-data; every durable NRP output is under
/data/cbl1016/. Only kai-data and transient emptyDir caches are mounted (Kai, 2026-10-19,
notes/build-todo.md). No W&B credential is requested: W&B remains offline on the PVC, project
bnjettag-chang-bitlinear, group chang-bitlinear-20261016 (STUDY §12).

Authority and accounting:
- 400 GPU-h total, with a ring-fenced 50 GPU-h reserve and at most four unfinished GPU Jobs,
  including pending Jobs (STUDY §10, Cap; harness concurrency semantics). Training has 350 GPU-h.
- The required screen has 51 runs: CH-8 3; F000/F100 6; six factorial configurations in two forms
  36; PID pair 6. All 66 including controls do not have automatic admission (STUDY §10 table).
  Required runs keep all seeds and 7000 epochs. Measured per-class all-inclusive rates reserve
  the required set first; there is no 1.10 multiplier on measured rates (STUDY §10 Formula/Admission).
- ADMISSION.json (initial template code/tools/admission.json) enumerates exact generated config paths: 51 required and 33 engineering
  identities, all still subject to check_launchable and runner support. Draws admit seeds 1–3,
  separately for record and smoke archives. No control or S4 launch is admitted. S4 options
  A/B/C add 20/32/16 runs and require Kai's separate approval (STUDY §10 table and cap).
- Initial tensors follow U1: one CH-8 draw per seed and archive on the product of record, one
  visible GPU, saved under initial/<product>/<record|smoke>. Smoke tensors never seed record runs.
- Every new Job reserves its deadline-based GPU bound. Engineering, failed-Job infrastructure
  retries and restarts debit the reserve; ordinary first production attempts debit training.
  The adapter accounts GPU allocation from pod status.startTime through the last init/application
  container termination, including init time (B17-12). Unknown use keeps its bound. Failed and
  pending Jobs remain in the ledger. No quota is invented.
- Kai's "if things fail or queue fails, just loop until it succeeds" (notes/build-todo.md,
  B17 correction) authorizes infrastructure relaunches without a count limit: eviction/preemption,
  node failure, image/volume mount failure, scheduling/pending timeout, and activeDeadlineSeconds
  exceeded while observed pending with no training start. Each must pass the 350/50
  GPU-h budget buckets and G80. Every existing run resumes its latest checkpoint through M14;
  absent/invalid checkpoints are refused, and verified completed siblings remain complete.
  Code/integrity exits, refused configs/resumes, hash/provenance mismatches and gate failures
  stop the lineage and preserve evidence for a new committed fix, checks and handoff (STUDY §11).
  Automatic agent repairs remain disabled (B17-2).
- BRIEF authority scientific_settings.b17_retry_policy sets infra_relaunches_max to null (no
  count limit) and infra_relaunch_notify_after_24h to 6. More than six successful infrastructure
  relaunch submissions of one pack in a rolling 24 h notify Kai through the approved local
  NOTIFY.log file and keep going. The threshold is a notification, never a stop or approval gate.
  The protected map carries this operational policy solely for compatibility with the unchanged
  lab schema. Legacy top-level retry caps are zero: direct lab-harness retry is disabled; the
  campaign adapter implements unlimited retry decisions. No scientific recipe changes.
- Every wave and every resubmission checks namespace-wide kai ConfigMaps against distinct
  ConfigMaps mounted by unfinished kai Jobs. In bundle-only operation both must be zero.
  A mismatch refuses submission; the cluster owner reconciles it (§11 ConfigMaps; G80).

Operations:
Use `code/tools/campaign.py --lab-root <lab> submit <campaign> <handoff>` and the same adapter's
`tick <campaign> --holder cron`. It uses the unchanged lab harness's approval, ledger, lifecycle
and retry logic, adding identity, split-budget and G80 checks to every submit, including tick's
relaunch path. Direct lab-harness submit/tick bypass those added checks and is outside this BRIEF.
The adapter replaces the later-stage-only reserve calculation with the two simultaneous buckets
above; both use 400/50 from §10. It does not install an admission controller.

Initial prepare: from code/, `python3 -B -m tools.operations prepare --campaign-dir <campaign>
--spec <pack.json> --bundle <bundle> --bundle-path /data/cbl1016/code/<name>.bundle --commit <sha>
--data-info <data_info.json> --launcher <lab>/tools/run_handoff.py`.
Relaunch/evaluate/workspace/repair-check argv are protected authority keys below because the
current harness schema does not accept commands in the operations block. They prepare a new
Job, record lifecycle success without physics scores, verify the source checkout, and refuse
automatic code repair, respectively. Benchmark rates: `python3 -B -m tools.bench_rates <capture>
--lifecycle <captured-lifetime.json>`. Evaluation does not open the held-out archive.

Operations numbers not specified by STUDY: lease 3600 s (B17-3, local scheduling default),
notification threshold 6 relaunches per 24 h (Kai B17 correction, B17-1). Disabled capabilities use 0; factors use 1. The unused score deadline 600 s is
the harness schema minimum (B17-3). `deadline_seconds_per_epoch` 3 is STUDY §10's planning rate,
not a measured deadline: the Job generator requires explicit resource/deadline inputs and never
uses it. No borrowed paused-campaign timing or resource estimate certifies this workload.

All generated handoffs stay pending until gate references and authority are reviewed. Production
certification is null in ADMISSION.json: B19 must supply product/K-specific measured rates,
required-screen projection, rolling three-hour utilization, device/host memory and product gates.
Updating the admission file changes its frozen hash and requires a new authority approval.
ADMISSION.json stays outside the code bundle, so it can bind B18's final commit without a
self-referential commit hash. The code/tools template is never silently rewritten.

'''
    return prose + '```json authority\n' + json.dumps(auth, indent=1, sort_keys=True) + '\n```\n\n' + \
        '```json operations\n' + json.dumps(ops, indent=1, sort_keys=True) + '\n```\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--out', required=True)
    p.add_argument('--admission', help='reviewed campaign ADMISSION.json; defaults to unapproved template')
    a = p.parse_args(argv)
    Path(a.out).write_text(draft(a.code_root, a.admission))


if __name__ == '__main__':
    main()
