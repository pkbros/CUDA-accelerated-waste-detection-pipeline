# Demo script — 80 second walkthrough for a fresher CUDA engineer interview

Plain English, one idea per section, two sentences maximum per section.
This is a prototype: every number on screen is read from a CSV at render time, and the video is rendered offline.

## 0. Footage checklist — do this before you record
- [ ] If real dataset val frames are available, use them and say "real dataset val frames".
- [ ] If they are not available, the footage is stitched from still frames — say that instead, and do not call it belt video.

## 1. Title — where does the time go? (0-8 s)
This is a CUDA prototype, not a shipped system.
The question is simple: where does the time go in a conveyor waste detector?

## 2. Split-screen — same frames, two pipelines (8-35 s)
Both pipelines run inference on the GPU; the difference is preprocessing, backend and pipelining.
The left panel is the baseline and the right panel is the optimized path, on the same frames, and the overlay shows measured per-frame timings, video rendered offline.

## 3. Overview panels B / C / D (35-50 s)
Panel B is the vector-add bandwidth ceiling, C is the inference ladder, and D is multi-feed batching.
Each panel has one caption line taken from the CSV, and any value that was never measured reads "result pending".

## 4. Nsight Systems timeline (50-62 s)
This is the profiler timeline, not a claim copied from a datasheet.
Caption: copies and kernels overlapping across streams.

## 5. What I got wrong (62-75 s)
FP16 lost to FP32, float4 did not beat scalar loads, and the pipelined loop regressed at two and four feeds.
Those numbers come from infer.csv, vector_add.csv and batching.csv, and they stay in the demo on purpose.

## 6. Repo and summary (75-80 s)
The repo URL on screen is read from the git remote, not typed in by hand.
Closing line: a measured CUDA prototype with the regressions left visible.

## Render
After the Colab benchmark writes results/demo_frame_timings.csv, run `python demo/render_demo.py`.
It writes demo/demo.mp4 and a 15 second demo/demo_short.mp4, and it needs ffmpeg on PATH.
