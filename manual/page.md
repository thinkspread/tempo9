# The built-in page

A server started with a vision encoder also serves a page at its own
address:

```
tempo9 --hf unsloth/Qwen3.5-9B-GGUF:Q4_K_S --tower auto
[tempo9] http://127.0.0.1:11435/v1 as Qwen3.5-9B-Q4_K_S.gguf
[tempo9] page: http://127.0.0.1:11435/ -- ask about an image, or keep an eye on a camera or screen
```

Open it in any browser on the same Mac. It has two tabs.

## Ask

Paste a screenshot (⌘V), drop an image, choose a file, or take one from the
camera or the screen, then ask. The answer streams in. Under it the page
shows the time to the first token, decode speed, the prompt's token count
(the image is most of it) and the image size sent. Images are scaled to at
most 1440 px on the long side; past about one megapixel the encoder scales
them down anyway.

For example, on an M5 Pro 24 GB with Qwen3.5-9B Q4_K_S, a 900×560 sign with
English and Chinese text was transcribed exactly. The prompt was 536 tokens,
the first token came after 1.33 s, decode ran at 48 tok/s, and the whole
answer took 2.6 s.

## Keep an eye on

Pick a camera, a screen or window, or a video file. Then write **rules**:
what to look for, what to say, and what to do.

| When it sees… | Say… | Do |
|---|---|---|
| a build or test run has finished successfully | The build is done. | beep + notify |
| a person is standing at the door | Someone is at the door. | beep + notify |

Each time the model is asked, it gets the numbered list and answers with the
first rule that holds, or 0. Order is priority, so put specific rules
(`a person holding a parcel`) above general ones (`a person`). A rule
acts when it becomes the answer, not for as long as it stays the answer. A
visitor is greeted once, and the rule is armed again after an answer of 0.
The first look after **Start** acts like any other, so a rule that already
holds then acts at once. While a rule holds, the status line under the
chart names it, so you can see what the page is waiting for.
The **say** line is read aloud with the Mac's own voices, picking a Chinese
voice when the line has Chinese in it. The rules are kept in this browser
between visits.

Presets: *a build finishes or fails* and *someone comes to the door*.

### How few frames reach the model

1. Once a second, the page compares a 32×18 thumbnail with the last frame it
   measured. A screen that has not changed stops here.
2. Otherwise the frame goes to `POST /v1/vision/frame`. That route runs only
   the vision encoder and returns one vector for the whole frame, plus one
   for each of 12 regions (a 4×3 grid). The page measures how far each
   region has moved from the frame the model last saw and takes the
   largest. A whole-frame vector dilutes a local change by its area: with
   the centre 10 % of a camera frame replaced it moved only 0.009, and 30 %
   moved 0.023. Either way, someone filling less than a third of the
   picture would never have woken the model.
3. The model is asked only when the frame has moved past **fire** and then
   held still below **settle**. With *react at the first sign of change*
   ticked, it is asked as soon as the frame moves past **fire**, for when
   seconds matter more than a still picture.
   That early look can catch someone halfway out of the frame. So when the
   frame then stops moving somewhere the model has not seen, the model is
   asked once more. That second look sees the empty doorway and re-arms the
   rules.

A **backstop** check every 60 s catches changes too small for the gate. It is
skipped when the frame has not changed at all.

The thresholds follow the scene's own noise, as Eyes On's do: **fire** is
3.5× a running average of the frame-to-frame distance while nothing
happens, **settle** 2×, each clamped to a range that depends on the source:

| Source | fire | settle |
|---|---|---|
| camera | 0.030–0.060 | 0.020–0.040 |
| screen, video file | 0.010–0.040 | 0.006–0.020 |

A screen has no sensor noise (an unchanged frame measures ~5×10⁻⁷), so its
range sits lower. Measured on Qwen3.5-9B's encoder, as the largest region
distance: sensor noise 0.013–0.023, a 4 % exposure change 0.007, a patch
covering 5 % of the frame 0.076–0.096, 10 % 0.11–0.12. A 2 % patch
(0.035–0.088) does not always fire.

A server without region vectors gets the whole-frame vector and fixed
thresholds: 0.025 / 0.010 for a camera (the Eyes On app's measurements: a
real scene change moved the vector 3–9× as far as sensor noise, and 0.025
passed ~5 % of frames with no false alarms) and 0.008 / 0.004 for a screen
or video file. "BUILD SUCCEEDED" appearing in a dark terminal measured
0.017 there, under the camera threshold, and with it the change waited 70 s
for the backstop.

With the whole-frame screen thresholds, before region vectors, a synthetic
build run (a progress bar for 25 s, then "✓ BUILD SUCCEEDED") went like
this:

- 34 frames were sampled.
- The encoder ran on 10 of them, the model on 2: the first look, and once
  after the success line appeared.
- The alert came about 2 s after the line appeared.
- The Chinese page with the Chinese preset gave the same counts (35 frames,
  10 encoder calls, 2 model calls).

The counters above the chart show the encoder and model counts as it runs.
With region vectors, the **fire** and **settle** boxes show the live
thresholds and cannot be edited; the backstop can.

The encoder runs on the GPU. A 640 px gate frame took 0.23 s with
Qwen3.5-9B's encoder (median of 20; 0.06–1.6 s). The same frames on the
Neural Engine took 2.1 s, so the gate stays on the GPU. The two compute
units give the same vector (cosine 0.9999 between them for the same frame),
so the camera thresholds, which were measured on the Neural Engine, hold on
the GPU too.

Keep the page visible while it watches. The page keeps its 1 s pace in a
background tab, but a browser may stop delivering new frames from a hidden
tab's video.

## Language

The page follows the browser's language, English or Chinese. The
**中文 / EN** button switches it, and the choice is remembered. Presets come in
both languages, and rules you wrote stay as you wrote them. On the Chinese page
the model is asked to explain its answer in Chinese, and a line with Chinese in
it is read with a Chinese voice.

## What it can reach

The page is served from the API's own origin, so it needs no CORS, and the
server sends none. Its Content-Security-Policy allows connections to that
origin only (`connect-src 'self'`). The page loads nothing else: no script,
font, sound or image from anywhere; the beep is synthesised. Camera and screen access are the browser's own
permission prompts.

## `POST /v1/vision/frame`

Not an OpenAI route. It is what the watch gate calls, and it is documented
here because it is useful on its own.

```
POST /v1/vision/frame
{"image": "data:image/jpeg;base64,..."}
```

```
{"object": "vision.frame", "dim": 4096, "tokens": 280, "encode_ms": 481.8,
 "embedding_f32": "<base64 little-endian float32[dim]>",
 "regions_f32": "<base64 little-endian float32[12 * dim]>",
 "region_grid": [3, 4]}
```

The vector is the encoder's output for the image, averaged over its tokens and
scaled to unit length, so the distance between two frames is `1 - dot(a, b)`.
`regions_f32` holds one more such vector for each cell of `region_grid`
(rows, columns) over the encoder's token grid, row-major. The watch gate takes
the region that moved most, since a whole-frame vector dilutes a small change
by its area. A server without them omits both fields.
The frame is not added to the encoder's cache, and the language model does not
run. Like chat, it takes `data:` URIs only, and it returns 400 when no vision
encoder is loaded.
