# liveclip: clip a live stream while it's still live

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aadi1659/liveclip/blob/main/liveclip_colab.ipynb)

Most clippers download the whole stream after it ends, then spend hours looking for moments. By then the first clips are already up and have taken the views.

liveclip works **during** the stream:

1. **Streamlink** records the live stream from the second you start it.
2. **A script reads the live chat** (YouTube or Twitch). No login and no API key are needed.
3. **When the chat suddenly explodes**, that moment is a highlight. **FFmpeg** cuts the last ~30 seconds into a vertical 9:16 clip.
4. **Whisper** (it runs on your own computer) adds captions.

All four tools are free and open source. The catch: it only works while the streamer is live.

> **Only clip streamers who allow clipping** (or who run an official clipping program). Clips without permission get copyright strikes and no payout.

---

## 💻 On your laptop: the clip app (easiest)

Paste a video link, mark the streamer's face and the gameplay, get vertical clips with captions.

1. Download this repo (green **Code** button → **Download ZIP**) and unzip it.
2. **Mac:** double-click **Start liveclip**. The first time it sets itself up (a few minutes); after that it opens straight away.
3. Your browser opens the liveclip page. Paste a YouTube or Twitch video link (or drop a video file) → **Get preview**.
4. Pick **Face box**, click two corners around the streamer's face. Pick **Gameplay box**, click two corners around the game.
5. Choose the number of clips, clip length, face size and caption style → **Make clips**. Clips show up below and are saved in the `clips` folder.

It finds the moments from loudness peaks (shouting, hype, big reactions). For live streams, use the live clipper below.

---

## 📱 On your phone (free, no install)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aadi1659/liveclip/blob/main/liveclip_colab.ipynb)

1. Tap **Open in Colab** above and sign in with any Google account.
2. Paste the live stream link in the box (YouTube app: **Share → Copy link**) and tap the **▶ play button**. If Colab says *"This notebook was not authored by Google"*, tap **Run anyway**.
3. Keep the tab open. Each clip shows up under the box as the chat spikes, with a **Download** button.

It runs on Google's free Colab computers, so your phone only shows the results. Free Colab sessions stop after a few hours or when idle. For faster captions, pick **Runtime → Change runtime type → T4 GPU** first.

---

## On your laptop: setup (one time, ~10 minutes)

You need **Python 3.10-3.12** and **FFmpeg**.

**Mac**
```bash
brew install python@3.11 ffmpeg
```
**Windows:** install Python 3.11 from python.org (tick "Add to PATH"), and FFmpeg with `winget install ffmpeg`.

Then download this repo and set it up:
```bash
git clone https://github.com/Aadi1659/liveclip.git
cd liveclip
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

```bash
python liveclip.py "https://www.youtube.com/watch?v=LIVE_VIDEO_ID"
```
- Leave it running while the stream is live. Clips land in `clips/` as `clip_<date>_<time>.mp4`, with a matching `.srt` caption file.
- **Press Enter** to clip the last ~30 seconds by hand.
- **Ctrl+C** to stop. Clips that are still being made get finished first.
- `clips/clips.csv` logs every clip: when it was made, how fast the chat was and the top chat words.

The first time you run it, Whisper downloads its model (~1.5 GB for `medium`) while the stream records, so start it a few minutes early.

## Useful options

| Option | What it does |
| --- | --- |
| `--pre 25 --post 6` | Seconds kept before/after the chat spike. Chat reacts a few seconds late, so most of the clip comes *before* the spike |
| `--spike 3` | How sudden the spike must be (3 = chat 3x faster than normal). Lower it for more clips, raise it for fewer |
| `--cooldown 60` | Minimum seconds between two auto clips |
| `--captions hinglish` | Hindi/English speech in Roman letters, the way people type it (default). `english` = translated to English, `off` = no captions |
| `--model medium` | Whisper size. `medium` (default) handles Hinglish well, about 10 s per clip. `small` is faster but garbles Hinglish |
| `--layout crop` | Center-crop to 9:16 instead of the blurred-background layout |
| `--max-clips 10 --duration 3600` | Stop after 10 clips or after 1 hour (the last clip is still finished) |
| `--start-time 00:30:00` | Wait until delay or scheduled start time before recording |
| `--max-time 24h` | Maximum stream duration to record/process (supports up to 24 hours: e.g. 24h, 120m, 3600s) |
| `--device cuda` | Run Whisper on a GPU. `auto` (default) uses one when it's there |

## Use it with Claude Code

Open the `liveclip` folder in Claude Code and just say:
> "Clip this stream: https://www.youtube.com/watch?v=..."

Claude reads `CLAUDE.md`, sets everything up and runs it for you. You can also ask it to change things ("make clips 45 seconds", "only clip when chat says W or LOL").

## Honest notes
- A chat spike means *something* happened. It doesn't guarantee a good clip, so watch each clip before you post it.
- Big streams (50K+ viewers) already have fast chat, so it rarely triples. If you get no clips, lower `--spike` to 1.7 (what the phone version calls "normal"); too many, raise it.
- Captions are auto-generated and can have mistakes. Fix them in Edits/CapCut, or edit the `.srt` file.
- Clipping programs pay per view and usually have caps. Read each campaign's rules before you post.
