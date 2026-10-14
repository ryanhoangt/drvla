# RQ2: paired SAE intervention pilot

Mục tiêu đầu tiên: kiểm tra chỉnh một nhóm SAE feature có thực sự đổi action,
đổi đối tượng robot thao tác, và có cứu được rollout task-perturbation không.
Đây là bước hướng đến steering cải thiện SR, **chưa phải bằng chứng rằng một
feature đã được xác nhận là memorization**.

Đã chạy 72 raw-case + 36 env-canonical rollouts. Đọc
[báo cáo tổng hợp bằng tiếng Anh](RQ2_report.md) trước khi diễn giải SR:
targeted edits chưa cứu wrong-object traps, và một completion rescue cũng xuất
hiện với random/general controls.

## Code map

| File | Vai trò |
|---|---|
| `prepare_features.py` | Freeze ID-only target/control IDs, reference means và input hashes |
| `normalize_feature_prompts.py` | Tạo bundle mới đúng prompt env, không đổi feature sets |
| `interventions.py` | Coefficient edits, decoder norm matching, recipient error preservation |
| `serve_policy_interventions.py` | GPU policy server và forward hook |
| `rollout_logger.py` | CPU simulator, paired counterfactual gates, full rollouts và physical labels |
| `run_pilot.py` | Quản lý hai Python envs, owned processes, source snapshots và artifact audits |
| `analyze_rq2.py` | Paired outcome summary, control fidelity và per-run report |
| `make_figures.py` | Sáu report plots từ completed artifacts; không chạy lại policy |

## Report figures

```bash
env openpi/.venv/bin/python -m experiments.rq2.make_figures
```

Plots được lưu trong `workspace/rq2/figures/`, cùng `figure_data.json` và
`figure_manifest.json` ghi số liệu/checksum. Script đọc dữ liệu đã chạy,
kiểm tra input không thay đổi, không sửa per-run reports/logs/NPZ files.

## Thiết kế đã triển khai

- Policy: pi0.5 LIBERO, frozen; SAE deployment-compatible
  `saes/pi05_libero_server`, pooled `paligemma.layer_5.output`.
- Chọn top 16 feature theo Cohen's d của mỗi training task từ ID demonstrations,
  như RQ1; bỏ feature chung giữa task gốc và task được yêu cầu mới.
- Pilot `libero_object_task`: task IDs 1, 2, 4, 5; task 1 là reference từ RQ1,
  các task còn lại là candidate memory-trap. Các ID được chọn có chủ đích,
  không phải random sample của benchmark.
- Cùng simulator initial state, seed, observation ban đầu và diffusion noise
  theo `(task, episode, inference-call)` giữa mọi condition.
- Chỉnh sáu inference calls đầu = 30 executed actions; giữ horizon 280 và replan
  mỗi 5 actions giống RQ1. Không tuning trên kết quả pilot.

| Condition | Can thiệp |
|---|---|
| baseline | Không chỉnh; vẫn log SAE |
| suppress_old | Giảm exclusive old-task coefficients về zero khi alpha=1 |
| boost_new | Tăng exclusive new-task coefficients lên ít nhất ID task mean |
| patch_old | Chuyển old-task coefficients từ donor cùng image/state/noise, prompt gốc |
| random_suppress / random_boost | Feature control, không thuộc old/new target sets |
| general_suppress / general_boost | Control từ classifier P(general) >= 0.5 |
| old_prompt | Không chỉnh SAE; prompt và scoring đều theo task GỐC, positive control |

`patch_old` là prompt-conditioned mediation test, KHÔNG phải patch từ một
episode ID-layout. Task perturbation đổi yêu cầu, không nhất thiết đổi vị trí;
chưa có position-shift / old-location causal test trong pilot này.

## Vì sao can thiệp như vậy?

SAE ở repo chuẩn hóa từng activation trước khi encode. Với pooled activation
recipient `x`, scale `s(x)`, và coefficient delta `df`, dùng:

```text
x' = x + s(x) W_dec df
H'[:, token, :] = H[:, token, :] + s(x) W_dec df
```

Giữ nguyên recipient normalization và reconstruction error; không thay toàn bộ
hidden state bằng SAE reconstruction. Broadcast cùng delta qua tokens giữ các
token differences ban đầu (ngoại trừ BF16 rounding). SAE re-encoding sau chỉnh
không nhất thiết đúng `f + df`: decoder directions không orthogonal và TopK có
thể đổi support. Cả intended và re-encoded coefficients đều được log.

Tutorial `SAE_Interp_Tutorial.md` hữu ích cho SAE error term, zero-edit hook
verification và so sánh actual ablation với attribution approximation. Paper
`2603.19183v2.pdf`, §3.4 / Appendix F.2, dùng decoder-direction broadcasting;
projection ablation trong paper khác coefficient ablation đang dùng ở đây.

## Controls và giới hạn

Fixed control feature IDs được match ID activation mean/frequency trước rollout.
Decoded perturbation norm sau đó được match tại từng recipient observation.
Nếu fixed suppression set không đủ năng lượng, chọn top-active cùng số lượng
trong candidate pool đã freeze; log actual IDs. Vì thế đây là adaptive matched
control, không phải fixed random set thuần túy xuyên suốt trajectory.
Không cho phép latent âm; nếu general pool không đủ năng lượng, control norm
bị cap. Report ghi rõ tỷ lệ match; không suy luận feature specificity nếu norm
matching kém. PG5 có rất ít general features theo classifier cũ.

Task-feature IDs được chọn hoàn toàn từ ID demonstrations nhưng không dùng
held-out scoring. Phân biệt task selectivity với episode-specific memorization;
thử nghiệm này không thay thế kiểm định memorization chặt hơn trong methodology.

## Chạy

Policy/analysis dùng Python 3.11 trong `openpi/.venv`; simulator dùng environment
LIBERO Python 3.8. Runner quản lý cả hai, bind localhost và chỉ dừng server nó
tự tạo. Tối đa hai simulator workers cho allocation 8 CPU cores / 1 H100.

```bash
env openpi/.venv/bin/python -m unittest experiments.rq2.test_interventions -v

env openpi/.venv/bin/python -m experiments.rq2.prepare_features \
  --out workspace/rq2/features_pg5_k16

env openpi/.venv/bin/python -m experiments.rq2.normalize_feature_prompts \
  --features workspace/rq2/features_pg5_k16/features.json \
  --out workspace/rq2/features_pg5_k16_env

env openpi/.venv/bin/python -m experiments.rq2.run_pilot \
  --run-dir workspace/rq2/my_preflight --tasks 2 --episodes 0 \
  --workers 1 --verify-only

env openpi/.venv/bin/python -m experiments.rq2.run_pilot \
  --run-dir workspace/rq2/my_pilot --tasks 1 2 4 5 --episodes 0 1 \
  --workers 2

env openpi/.venv/bin/python -m experiments.rq2.analyze_rq2 \
  --run-dir workspace/rq2/my_pilot
```

Full pilot mặc định: 4 tasks × 2 initial states × 9 conditions = 72 rollouts.
Runner mặc định dùng `features_pg5_k16_env/features.json`; logger assert prompt
khớp chính xác language của simulator. Run đầu dùng raw BDDL capitalization,
được giữ riêng để audit; đây không phải bản tái hiện chính xác prompt protocol RQ1.
Case-canonical confirmation giữ nguyên feature IDs/means và chỉ đổi prompt text.
`--video` lưu mp4; raw input images đã được lưu cho mọi inference call dù không
bật video. `--intervene-calls -1` chỉnh toàn episode; phải dùng run directory mới.
Thay layer/K/alpha/window cần run directory riêng và feature bundle tương ứng.

## Gate và artifacts

Trước mỗi paired block, thực hiện cùng-observation counterfactuals cho tất cả
conditions và assert action equality tuyệt đối cho unhooked, baseline, identity,
và repeated baseline với cùng noise. Failure của gate dừng worker.

Chỉ ghi bên trong `workspace/rq2/`; manifest kiểm tra cấu hình/source hashes
khi resume. Hoàn thành từng rollout bằng atomic NPZ write rồi append result.
Resume chỉ bỏ qua rollout đã có record và artifact. Không sửa SAE/checkpoint,
activation/index, source RQ1 hay kết quả RQ1. Audit size/mtime của các input này
trước/sau run; hash các input nhỏ quan trọng trong feature manifest.

```text
workspace/rq2/<run>/
  run_manifest.json
  clients/tasks_<id>.json
  verification/task<id>_ep<id>_seed<seed>.{json,npz}
  rollouts/task<id>/
    results.jsonl
    ep<id>_seed<seed>_<condition>.npz
    ep<id>_seed<seed>_<condition>_initial.npz
  logs/{server,task<id>}.log
  protected_artifacts_audit.json
  summary.json
  RQ2_report.md
```

Success đánh giá toàn bộ new-goal predicates sau action, kể cả action cuối;
`old_prompt` exception đánh giá old goal. Old-task event và terminal trap failure
là hai nhãn riêng: event không đồng nghĩa failure. First lift chỉ là proxy,
được lưu cùng object positions, end-effector positions và predicate timelines
để kiểm tra trực tiếp; không dùng thời điểm lift làm failure onset.

Next gate khoa học: target edit cần tạo outcome khác matched random/general
controls, không chỉ đổi action. Nếu pilot null, kiểm tra edit magnitude/support,
causal layer, token specificity và dose/window trước khi kết luận SAE không giúp.
Nếu có rescue, cần held-out tasks/states, dose-response và kiểm tra ID SR trước
khi claim steering giảm memorization hoặc tăng SR toàn LIBERO-PRO.
