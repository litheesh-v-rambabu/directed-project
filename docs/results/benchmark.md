Model `fused_speaker_s0` (552,771 parameters). Intel(R) Xeon(R) Processor @ 2.10GHz, 1 thread, torch 2.15.0+cu132, 300 timed batch-1 calls on test utterances (model forward only; log-mel extraction for 2.5 s of audio takes 0.57 ms p50). Accuracy and FRR/FAR use the FP32 model's calibrated 5%-FAR threshold for every backend.

| Backend | Size MB | p50 ms | p95 ms | Throughput utt/s (batch 32) | Accuracy % | FRR % | FAR % | EER % | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 2.24 | 2.57 | 4.13 | 742 | 91.80 | 10.59 | 5.66 | 7.31 | 0.9784 |
| INT8 | 0.88 | 2.92 | 4.46 | 868 | 91.80 | 10.50 | 5.77 | 7.31 | 0.9784 |
| ONNX | 2.33 | 1.74 | 2.83 | 473 | 91.80 | 10.59 | 5.66 | 7.31 | 0.9784 |
