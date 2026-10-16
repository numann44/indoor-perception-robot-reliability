[Watch the public demo](https://numann44.github.io/indoor-perception-robot-reliability/)

# Controlled simulation demo

![Simulation preview](../assets/simulation-preview.gif)

[Download/view the 78 s project walkthrough](portfolio-demo.mp4) · [Video player source / offline page](index.html) · [Frame metadata and provenance](simulation-manifest.json)

The two recorded MuJoCo chapters each span 23 seconds at original 1× source time. Intro/results/limits cards extend the walkthrough to 78 seconds. Displayed panels are saved selected-model outputs sampled from 230 processed sensor frames per chapter, not newly generated illustrations. Between samples the video holds the previous observed panel. Source timestamps, missing depth and commanded stop remain visible.

These media chapters use geometric control while selected learned detection/semantic/tracking outputs are displayed as shadow observations. They do not establish learned-control causality. Separate live ROS2/matched episode results are summarized in docs/RESULTS.md. Simulation and recorded robot datasets do not establish candidate-operated physical robot or embedded target validation.

GitHub displays the GIF directly and offers the MP4 download without a localhost dependency. GitHub shows HTML as source; to use the offline video page, download the repository ZIP and open index.html alongside its MP4. The full embedded source-panel viewer remains local; the public subset contains derived MP4/GIF plus panel hashes and metadata, not individual PNGs.
