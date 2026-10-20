# B17 harness integration

The lab-root harness and launcher remain unchanged. Install the reviewed BRIEF.draft.md as the
campaign BRIEF.md and ADMISSION.draft.json as ADMISSION.json, outside code/. Regenerate frozen
hashes with `python3 -B -m tools.draft_brief --out <draft> --admission <ADMISSION.json>` after
merging code. Kai approves only after gates; no approval is written by B17.

Use `python3 code/tools/campaign.py --lab-root LAB submit CAMPAIGN HANDOFF` for every initial
wave and `... tick CAMPAIGN --holder cron` for automatic work. The adapter installs additional
checks on the unchanged harness's submit function, including calls made by tick. It preserves
the harness's approval, frozen-file, lifecycle, state lock, ledger and hard-bound checks. The
campaign split budget supersedes the harness's later-stage reserve calculation: first production
attempts use at most 350 GPU-h; engineering, failed attempts and retries use at most 50 GPU-h.
The adapter extends the harness's per-pod accounting from status.startTime through the last
init/application container termination, so measured use includes init time. Missing lifecycle
evidence retains the full bound. Four unfinished Jobs include pending and ambiguous submissions;
terminal Jobs without complete lifecycle evidence conservatively retain a slot until reconciled.

Before each submission the adapter refreshes lifecycle and checks G80 against all kai Jobs in
cms-ml, not just this campaign. ConfigMap names come from metadata only. Distinct mounted names
must equal the complete kai- inventory, and both must be zero for bundle operation. A discrepancy
blocks the wave; no ConfigMap or Job is deleted automatically. Offline G80 checks use
`python3 -m tools.configmaps --configmaps SAVED_CM_METADATA --jobs SAVED_JOBS`.

Run all state-writing operations on the home PC. B17 installs no cron. Proposed line, to be
installed by the PC owner only after approved deployment (pattern from paused SETUP.md §3):

```
*/15 * * * * cd /home/kaimoe/lab/bnjettag && PATH=/home/kaimoe/.local/bin:/usr/bin:/bin /usr/bin/python3 campaigns/2026-10-16-chang-bitlinear/code/tools/campaign.py --lab-root /home/kaimoe/lab/bnjettag tick /home/kaimoe/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear --holder cron >> campaigns/2026-10-16-chang-bitlinear/cron.log 2>&1
```

The protected command contracts are in tools.operations: prepare, relaunch, evaluate, workspace,
repair-check. Their argv live in the authority block because the current harness schema requires
that location; adjustable scheduling settings live in operations. Relaunch only accepts the
evidence-backed infrastructure classification, preserves scientific identity and creates a new Job.
Kai's instruction in campaign notes/build-todo.md supersedes B17's two-relaunch default:
infrastructure failures loop without a count limit while the 350/50 GPU-h buckets admit the
relaunch and G80 passes before every wave. Causes include eviction/preemption, node failure,
image/volume mount failure, scheduling/pending timeout and activeDeadlineSeconds exceeded while
observed pending without a training start. Existing runs resume their latest M14 checkpoint;
missing/invalid checkpoints are refused and verified completed siblings are skipped.
Code/integrity nonzero exits, config/resume refusals, hash/provenance mismatches and gate failures
stop the lineage, preserve evidence and require a new committed fix, checks and handoff (STUDY §11).
Code/integrity evidence takes priority even when disruption is also reported; the monitor retains
terminal evidence after pod records disappear. Automatic repairs remain disabled.

The BRIEF field `scientific_settings.b17_retry_policy.infra_relaunches_max` is null (unlimited),
and `infra_relaunch_notify_after_24h` is 6. The protected map carries operational policy for
compatibility with the unchanged lab schema; it changes no scientific recipe. The adapter uses
unbounded retry decisions; both legacy top-level retry caps are zero to disable direct lab-harness
retries. More than six distinct successful infrastructure relaunch submissions per pack in a
rolling 24 h append a notification for Kai to the approved local `NOTIFY.log` and keep going.
Replayed submissions do not count twice; other packs and submissions at least 24 h old do not
count toward this threshold. Budget/concurrency deferrals do not count as relaunches.
 Evaluation records lifecycle success as exploratory engineering evidence; no model
metric, held-out read or scientific verdict is inferred.

ADMISSION.json explicitly lists the 51 required and 33 engineering generated-config identities.
check_launchable and runner support still apply to every identity. Controls and S4 remain excluded.
Production certification is initially null. It must bind final code_sha, gpu_product and k; status
passed; utilization_pass, device_memory_pass, host_projection_pass, fingerprint_pass and cost_pass
true; required_screen_runs 51; required_screen_gpu_h at most 350; and evidence references. These
are reviewed B19 measurements, not fields B17 fabricates. Its frozen hash is authority-bound;
changing it requires renewed approval, without a self-reference in the committed code bundle.

Pattern sources and full sha256 values are in pattern-sources.json. All reuse is conceptual;
paused-campaign scientific rules, rates, credentials and ConfigMap transport were not copied.
