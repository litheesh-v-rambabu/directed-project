Per-condition error analysis for `fused_speaker_s0` (test split, 5% FAR threshold).

**snr**

| Bucket | n | Accuracy % | FAR % | FRR % |
|---|---:|---:|---:|---:|
| <0dB | 79 | 93.7 | 4.3 | 20.0 |
| 0-8dB | 440 | 91.8 | 2.9 | 20.3 |
| 8-16dB | 792 | 90.7 | 7.4 | 11.3 |
| 16-24dB | 562 | 93.4 | 7.4 | 6.2 |
| >=24dB | 127 | 90.6 | 4.0 | 10.8 |

**duration**

| Bucket | n | Accuracy % | FAR % | FRR % |
|---|---:|---:|---:|---:|
| <1s | 5 | 100.0 | 0.0 | 0.0 |
| 1-2s | 594 | 87.4 | 7.4 | 17.0 |
| 2-3s | 1031 | 93.3 | 4.3 | 9.6 |
| >=3s | 370 | 94.6 | 8.1 | 4.1 |

**reverb_rt60**

| Bucket | n | Accuracy % | FAR % | FRR % |
|---|---:|---:|---:|---:|
| <0.25s | 573 | 94.4 | 20.9 | 3.6 |
| 0.25-0.5s | 976 | 89.7 | 6.6 | 14.3 |
| >=0.5s | 451 | 93.1 | 2.0 | 50.0 |

**speaker**

| Bucket | n | Accuracy % | FAR % | FRR % |
|---|---:|---:|---:|---:|
| enrolled user | 1299 | 90.3 | 5.9 | 11.3 |
| other speaker | 701 | 94.6 | 5.5 | 5.0 |

**Top false accepts**

| Transcript (ASR-style) | P(directed) | SNR dB | Enrolled speaker |
|---|---:|---:|---|
| play some rock | 0.986 | 5.8 | True |
| to the volume up | 0.976 | 14.0 | True |
| an alarm for noon | 0.973 | 11.2 | True |
| what did sarah say | 0.967 | 14.1 | True |
| i was wondering how long it takes to get to the airport | 0.966 | 17.9 | False |
| tell me a joke | 0.965 | 19.5 | True |
| set an alarm for eight thirty | 0.965 | 16.3 | False |
| send a message to dad saying running late | 0.962 | 15.6 | False |

**Top false rejects**

| Transcript (ASR-style) | P(directed) | SNR dB | Enrolled speaker |
|---|---:|---:|---|
| are you apples today | 0.002 | 3.4 | True |
| skip this song | 0.025 | 9.9 | True |
| can you turn hallway lights off | 0.028 | 2.2 | True |
| remind tomatoes to water the plants at midnight | 0.047 | 17.2 | True |
| set timer podcast forty minutes | 0.055 | 12.7 | True |
| what did mom say | 0.064 | 12.9 | True |
| skip this song | 0.09 | 13.9 | True |
| what time is it | 0.09 | 13.8 | True |
