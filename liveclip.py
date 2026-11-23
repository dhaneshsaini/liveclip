#!/usr/bin/env python3
"""liveclip: clip a live stream WHILE it is live, using the chat as the highlight detector.

  python liveclip.py <youtube-or-twitch-live-url> [options]

How it works
  1. Streamlink records the live stream from the moment you start (6-second .ts segments, nothing is re-encoded).
  2. A small script (chatread.py) reads the live chat. No login and no API key are needed.
  3. When the chat suddenly explodes (messages per second jump well above the stream's normal rate), that moment is a
     highlight. The script waits a few seconds, then FFmpeg cuts the last ~30 s into a 9:16 vertical clip.
  4. faster-whisper (OpenAI's Whisper, run locally) writes the captions, and they are burned into the clip.

While it runs: press Enter to clip the last few seconds by hand. Ctrl+C to stop.
Only clip streamers who allow clipping or run a clipping program. Clips without permission get copyright strikes.
"""
import argparse, csv, os, re, shutil, signal, statistics, subprocess, sys, threading, time
from collections import Counter, deque

ap = argparse.ArgumentParser(description="Clip a live stream when its chat spikes.")
ap.add_argument("url", help="YouTube or Twitch LIVE url")
ap.add_argument("--out", default="clips", help="output folder (default: clips)")
ap.add_argument("--quality", default="720p,720p60,480p,best", help="streamlink quality list")
ap.add_argument("--pre", type=float, default=25, help="seconds before the chat spike to include (chat reacts late)")
ap.add_argument("--post", type=float, default=6, help="seconds after the spike to include")
ap.add_argument("--spike", type=float, default=3.0, help="spike = chat rate this many times the normal rate")
ap.add_argument("--min-rate", type=float, default=1.5, help="ignore spikes below this many messages/second")
ap.add_argument("--cooldown", type=float, default=60, help="minimum seconds between two auto clips")
ap.add_argument("--warmup", type=float, default=60, help="seconds of chat to learn the normal rate before clipping")
ap.add_argument("--max-clips", type=int, default=0, help="stop after this many clips (0 = no limit)")
ap.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0 = run until Ctrl+C)")
ap.add_argument("--max-time", default="", help="stop after this duration (e.g. 24h, 120m, 3600s, or HH:MM:SS; up to 24 hours)")
ap.add_argument("--start-time", default="", help="start timestamp or delay before starting (e.g. 02:20:22, 10m, 1h, or HH:MM:SS)")
ap.add_argument("--end-time", default="", help="stop clipping at this timestamp in the video/stream (e.g. 05:20:12, 2h, or 24:00:00)")
ap.add_argument("--captions", choices=["hinglish", "english", "en", "hi", "auto", "off"], default="hinglish",
                help="hinglish (default): Hindi/English speech in Roman letters. english: translated to English. "
                     "en/hi/auto: plain Whisper language setting. off: no captions")
ap.add_argument("--model", default="medium", help="Whisper model: small (fast), medium (default, much better Hinglish), "
                                                  "large-v3 (best, slow)")
ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                help="where Whisper runs: auto uses a GPU when there is one (Colab GPU runtime), else the CPU")
ap.add_argument("--layout", choices=["blur", "crop"], default="blur", help="9:16 layout: blurred background or center crop")
ap.add_argument("--font", default="", help="path to a .ttf font for captions")
ap.add_argument("--keep", action="store_true", help="keep the raw recording segments when you stop")
a = ap.parse_args()

OUT = os.path.abspath(a.out); REC = os.path.join(OUT, ".rec"); os.makedirs(REC, exist_ok=True)
SEGLIST = os.path.join(REC, "segments.csv")
STOP = threading.Event()   # set when recording must end (stream offline, or everything is finished)
QUIT = threading.Event()   # set by Ctrl+C / limits: stop making new clips, but keep recording until the open ones are cut
LOG = os.path.join(OUT, "clips.csv")


def say(*m): print(time.strftime("%H:%M:%S"), *m, flush=True)


def parse_time_arg(val):
    if not val: return 0.0
    val = str(val).strip()
    if not val: return 0.0
    try:
        return max(0.0, float(val))
    except ValueError:
        pass
    parts = val.split(":")
    if len(parts) == 3:
        try: return max(0.0, float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]))
        except ValueError: pass
    elif len(parts) == 2:
        try: return max(0.0, float(parts[0]) * 60 + float(parts[1]))
        except ValueError: pass
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*([hms])', val.lower())
    if matches:
        mults = {'h': 3600, 'm': 60, 's': 1}
        return max(0.0, sum(float(n) * mults[u] for n, u in matches))
    return 0.0


def wait_for_start(start_val):
    if not start_val: return
    start_str = str(start_val).strip()
    if not start_str or start_str in ("0", "00:00:00", "00:00"): return
    parts = start_str.split(":")
    delay = 0.0
    if len(parts) in (2, 3):
        now_lt = time.localtime()
        target_h = int(parts[0])
        target_m = int(parts[1])
        target_s = int(parts[2]) if len(parts) == 3 else 0
        now_sec = now_lt.tm_hour * 3600 + now_lt.tm_min * 60 + now_lt.tm_sec
        tgt_sec = target_h * 3600 + target_m * 60 + target_s
        if tgt_sec > now_sec:
            delay = float(tgt_sec - now_sec)
        elif tgt_sec < now_sec and (now_sec - tgt_sec) > 3600:
            delay = float((86400 - now_sec) + tgt_sec)
        else:
            delay = parse_time_arg(start_str)
    else:
        delay = parse_time_arg(start_str)
    if delay > 0:
        say(f"Waiting {delay:.0f}s until stream starting time ({start_str})...")
        time.sleep(delay)


for tool in ("ffmpeg", "ffprobe"):
    if not shutil.which(tool): sys.exit(f"{tool} not found. Install FFmpeg first (Mac: brew install ffmpeg).")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def find_streamlink():
    which = shutil.which("streamlink")
    if which: return which
    for candidate in [
        os.path.join(os.path.dirname(sys.executable), "streamlink"),
        os.path.join(os.path.dirname(sys.executable), "streamlink.exe"),
        os.path.join(os.path.dirname(sys.executable), "Scripts", "streamlink.exe"),
        os.path.join(os.path.dirname(sys.executable), "bin", "streamlink"),
    ]:
        if os.path.exists(candidate): return candidate
    return "streamlink"
STREAMLINK = find_streamlink()

# ------------------------------------------------------------------ 1. record (streamlink -> ffmpeg 6 s segments)
rec_start = None


def record():
    global rec_start
    shutil.rmtree(REC, ignore_errors=True); os.makedirs(REC, exist_ok=True)   # leftovers from a force-stopped run
    if os.path.isfile(a.url):
        say(f"Processing local video file: {a.url}")
        t_s = parse_time_arg(a.start_time)
        t_e = parse_time_arg(a.end_time)
        cmd = ["ffmpeg", "-v", "fatal"]
        if t_s > 0: cmd += ["-ss", f"{t_s:.2f}"]
        if t_e > t_s: cmd += ["-t", f"{t_e - t_s:.2f}"]
        cmd += ["-i", a.url, "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                "-f", "segment", "-segment_time", "6", "-segment_list", SEGLIST, "-segment_list_type", "csv",
                os.path.join(REC, "seg_%06d.ts")]
        ff = subprocess.Popen(cmd, start_new_session=(sys.platform != "win32"))
        rec_start = time.time(); say("reading file segments started")
        while not STOP.is_set():
            if ff.poll() is not None:
                say("finished reading video file"); STOP.set(); break
            time.sleep(0.5)
        if ff.poll() is None: ff.terminate()
        return

    sl = subprocess.Popen([STREAMLINK, "--stdout", "--hls-live-edge", "2", a.url, a.quality],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=(sys.platform != "win32"))
    # Check if streamlink exits quickly (e.g. stream offline / ended VOD)
    time.sleep(1.0)
    if sl.poll() is not None:
        err = ""
        if sl.stderr:
            try: err = sl.stderr.read().decode("utf-8", "ignore").strip()
            except Exception: pass
        say(f"Live stream offline or ended ({err.splitlines()[-1] if err else 'ended stream'}). Streaming with yt-dlp...")
        t_s = parse_time_arg(a.start_time)
        t_e = parse_time_arg(a.end_time)
        ytdlp_cmd = [sys.executable, "-m", "yt_dlp", "-o", "-", "-f", "bv*[height<=720]+ba/b[height<=720]/b"]
        if t_s > 0 or (t_e > 0 and t_e < 86400):
            end_sec = t_e if (t_e > t_s) else (t_s + 86400)
            ytdlp_cmd += ["--download-sections", f"*{int(t_s)}-{int(end_sec)}", "--force-keyframes-at-cuts"]
        ytdlp_cmd += [a.url]
        try:
            sl = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  start_new_session=(sys.platform != "win32"))
        except Exception as e:
            say(f"yt-dlp fallback error: {e}")
            STOP.set()
            return

    ff = subprocess.Popen(["ffmpeg", "-v", "fatal", "-i", "pipe:0", "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                           "-f", "segment", "-segment_time", "6", "-segment_list", SEGLIST, "-segment_list_type", "csv",
                           os.path.join(REC, "seg_%06d.ts")], stdin=sl.stdout,
                          start_new_session=(sys.platform != "win32"))
    sl.stdout.close()
    while not STOP.is_set():
        if rec_start is None and os.path.exists(os.path.join(REC, "seg_000000.ts")):
            rec_start = time.time(); say("recording started")
        if ff.poll() is not None:
            err = ""
            if sl.poll() is not None and sl.stderr:
                try: err = sl.stderr.read().decode("utf-8", "ignore").strip()
                except Exception: pass
            if err:
                say(f"source: {err.splitlines()[-1]}")
            say("recording stopped (stream ended, video finished, or went offline)"); STOP.set(); break
        time.sleep(0.2)
    for p in (sl, ff):
        if p.poll() is None: p.terminate()
    for p in (sl, ff):
        try: p.wait(5)
        except subprocess.TimeoutExpired: p.kill()


def segments():
    """[(file, start, end)] of finished segments, in recording seconds."""
    if not os.path.exists(SEGLIST): return []
    out = []
    for row in csv.reader(open(SEGLIST)):
        if len(row) >= 3: out.append((os.path.join(REC, row[0]), float(row[1]), float(row[2])))
    return out


def rec_now():
    return 0 if rec_start is None else time.time() - rec_start


# ------------------------------------------------------------------ 2. chat (no login)
chat_times = deque(maxlen=200000)   # (wallclock arrival, text)


def chat():
    from chatread import read
    while not STOP.is_set():
        try:
            for msg in read(a.url):
                if STOP.is_set(): return
                chat_times.append((time.time(), msg))
        except Exception as e:
            if STOP.is_set(): return
            say(f"chat reconnecting ({e})"); time.sleep(5)


# ------------------------------------------------------------------ 3. spike detector
def per_second(now, span):
    """message counts per second for the last `span` seconds (oldest first)."""
    c = [0] * int(span)
    for t, _ in reversed(chat_times):
        k = int(now - t)
        if k >= span: break
        if k >= 0: c[int(span) - 1 - k] += 1
    return c


def top_words(t0, t1, n=4):
    words = Counter()
    for t, msg in chat_times:
        if t0 <= t <= t1:
            for w in set(re.findall(r"[\w']{2,}", msg.lower())): words[w] += 1
    return " ".join(w for w, _ in words.most_common(n))


# ------------------------------------------------------------------ 4. cut + captions
def font(size):
    from PIL import ImageFont
    cands = [a.font, "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
             "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
    for c in cands:
        if c and os.path.exists(c): return ImageFont.truetype(c, size)
    return ImageFont.load_default(size)


def caption_png(text, path, W=1080):
    from PIL import Image, ImageDraw
    f = font(64); d = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > W - 160 and cur: lines.append(cur); cur = w
        else: cur = t
    if cur: lines.append(cur)
    H = 90 * len(lines) + 40
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        x = (W - d.textlength(ln, font=f)) / 2
        d.text((x, 20 + i * 90), ln, font=f, fill=(255, 255, 255, 255), stroke_width=7, stroke_fill=(0, 0, 0, 255))
    im.save(path)


_whisper = None
_wlock = threading.Lock()
# With language="en", Whisper sometimes TRANSLATES Hindi speech into English. A natural Hinglish paragraph as the prompt
# makes it write what was said, in Roman letters ("yaha pe main helipad dhoondh raha hoon"). Tested on 5 windows of a
# Hindi GTA 6 stream, 19 Nov 2026: a short keyword prompt still translated the stream intro; this paragraph didn't.
HINGLISH = ("Nahi yaar, yaha se rasta hi nahi hai. Bhai yeh dekho, agar aap naye ho toh follow kar lo. "
            "Main abhi helicopter dhoondh raha hoon, kya scene hai, chalo chalo. Haan bhai, mast hai.")
HINGLISH_2 = ("Haan bhai, aaj ka stream mast hone wala hai. Yeh dekho kya ho gaya, maine socha nahi tha. "
              "Chalo mission start karte hain, tum log comment mein batao kaisa laga.")
MARKERS = set("hai hain nahi nahin yaar bhai kya ka ki ke ko se mein me pe main hoon raha rahi tha toh yeh ye woh wo aur "
              "bhi hi kar karo dekho".split())


def hinglish_share(words):
    ws = [re.sub(r"[^a-z']", "", w.word.lower()) for w in words]
    ws = [w for w in ws if w]
    return sum(w in MARKERS for w in ws) / max(1, len(ws))


def whisper():
    global _whisper
    with _wlock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            say(f"loading Whisper '{a.model}' (the first run downloads it once, ~1.5 GB for medium)")
            dev = a.device
            if dev == "auto":
                try:
                    import ctranslate2
                    dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                except Exception:
                    dev = "cpu"
            if a.model == "auto":                               # medium needs a GPU to be quick; small on a CPU
                a.model = "medium" if dev == "cuda" else "small"
                say(f"Whisper model: {a.model} ({dev})")
            try:
                _whisper = WhisperModel(a.model, device=dev, compute_type="float16" if dev == "cuda" else "int8")
            except Exception as e:
                if dev == "cpu": raise
                say(f"GPU Whisper failed ({e}), using the CPU"); dev = "cpu"
                _whisper = WhisperModel(a.model, device="cpu", compute_type="int8")
            say(f"Whisper ready ({dev})")
    return _whisper


def load_wav(path):
    """16 kHz mono 16-bit WAV -> float32 array. faster-whisper's own loader uses PyAV, and newer PyAV versions (Colab)
    break it ("unexpected keyword argument 'metadata_errors'"), so we hand Whisper the samples directly."""
    import wave, numpy as np
    with wave.open(path, "rb") as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav):
    mode = a.captions
    wav = load_wav(wav)
    kw = dict(word_timestamps=True, vad_filter=True, condition_on_previous_text=False, beam_size=5)
    if mode in ("hinglish", "en"):
        kw.update(language="en", initial_prompt=HINGLISH)
    elif mode == "english":
        kw.update(task="translate")
    elif mode == "hi":
        kw.update(language="hi")
    segs, _ = whisper().transcribe(wav, **kw)
    words = [w for s in segs for w in (s.words or [])]
    if mode in ("hinglish", "en") and words and hinglish_share(words) < 0.08:
        # looks like an English translation (or an English-speaking streamer): try once more, keep the more Hinglish one
        segs2, _ = whisper().transcribe(wav, **{**kw, "initial_prompt": HINGLISH_2})
        words2 = [w for s in segs2 for w in (s.words or [])]
        if hinglish_share(words2) > hinglish_share(words): words = words2
    chunks, cur = [], []
    for w in words:                      # up to 4 words per caption; break on pauses and sentence ends
        if cur and (len(cur) >= 4 or w.start - cur[-1].end > 0.6 or cur[-1].word.strip()[-1:] in ".?!,"):
            chunks.append(cur); cur = []
        cur.append(w)
    if cur: chunks.append(cur)
    merged = []                          # a lone word ("ho", "hogi") joins the caption before it
    for c in chunks:
        if merged and len(c) == 1 and len(merged[-1]) <= 4 and c[0].start - merged[-1][-1].end < 0.5:
            merged[-1] = merged[-1] + c
        else:
            merged.append(c)
    return [(c[0].start, c[-1].end, " ".join(x.word.strip() for x in c).strip(" ,.")) for c in merged]


_cut_lock = threading.Lock()   # one clip at a time: parallel encodes on a 2-core Colab machine all crawl


def cut(t0, t1, name):
    """t0/t1 in recording seconds -> OUT/name.mp4 (9:16, captions burned in)."""
    with _cut_lock:
        return _cut(t0, t1, name)


def _cut(t0, t1, name):
    deadline = time.time() + 60
    while not any(e >= t1 for _, _, e in segments()):          # wait until the recording covers the window
        if STOP.is_set(): break                                 # recording ended: cut what there is
        if time.time() > deadline: say("not enough recording for", name); return None
        time.sleep(1)
    segs = [s for s in segments() if s[2] > t0 and s[1] < t1]
    if not segs: say("not enough recording for", name); return None
    t1 = min(t1, segs[-1][2])
    tmp = os.path.join(REC, name); os.makedirs(tmp, exist_ok=True)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{f.replace(os.sep, '/')}'\n" for f, _, _ in segs))
    raw = os.path.join(tmp, "raw.ts")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", raw], check=True)
    ss, dur = max(0, t0 - segs[0][1]), t1 - t0
    if a.layout == "blur":
        # blur a 270x480 copy and scale it up: looks the same, ~10x cheaper than blurring at 1080x1920
        vf = ("[0:v]split[a][b];[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,"
              "scale=1080:1920,eq=brightness=-0.08[bg];[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
    else:
        vf = "[0:v]scale=-2:1920,crop=1080:1920[v0]"
    inputs, last = ["-ss", f"{ss:.2f}", "-t", f"{dur:.2f}", "-i", raw], "v0"
    if a.captions != "off":
        wav = os.path.join(tmp, "a.wav")
        subprocess.run(["ffmpeg", "-v", "fatal", "-y", "-ss", f"{ss:.2f}", "-t", f"{dur:.2f}", "-i", raw, "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
        try:
            caps = transcribe(wav)
        except Exception as e:
            say("captions skipped:", e); caps = []
        srt = []
        for i, (s, e, txt) in enumerate(caps):
            png = os.path.join(tmp, f"c{i}.png"); caption_png(txt, png); inputs += ["-i", png]
            vf += f";[{last}][{i + 1}:v]overlay=0:H*0.70:enable='between(t,{s:.2f},{e + 0.1:.2f})'[v{i + 1}]"
            last = f"v{i + 1}"
            srt.append(f"{i + 1}\n{ts(s)} --> {ts(e)}\n{txt}\n")
        if srt: open(os.path.join(OUT, name + ".srt"), "w").write("\n".join(srt))
    out = os.path.join(OUT, name + ".mp4")
    subprocess.run(["ffmpeg", "-v", "fatal", "-y", *inputs, "-filter_complex", vf, "-map", f"[{last}]", "-map", "0:a:0?",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def ts(s):
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(s % 1 * 1000):03d}"


# ------------------------------------------------------------------ main loop
def main():
    say(f"liveclip -> {OUT}")
    say("Only clip streamers who allow it. " + ("Enter = clip now, Ctrl+C = stop." if sys.stdin.isatty() else "Stop = finish open clips and quit."))
    if a.start_time: wait_for_start(a.start_time)
    rec_thread = threading.Thread(target=record, daemon=True); rec_thread.start()
    threading.Thread(target=chat, daemon=True).start()
    if a.captions != "off": threading.Thread(target=whisper, daemon=True).start()
    manual = threading.Event()

    t_start = parse_time_arg(a.start_time)
    t_end = parse_time_arg(a.end_time)
    max_dur = parse_time_arg(a.max_time) or a.duration
    if t_end > t_start:
        max_dur = min(t_end - t_start, 86400.0)
    elif t_end > 0:
        max_dur = min(t_end, 86400.0)
    elif max_dur > 86400:
        max_dur = 86400.0   # cap at 24 hours
    if max_dur:
        say(f"Max processing time set: {max_dur / 3600:.1f}h ({max_dur:.0f}s)")

    def keys():
        for _ in sys.stdin: manual.set()
    if sys.stdin.isatty(): threading.Thread(target=keys, daemon=True).start()

    def on_ctrl_c(*_):
        if QUIT.is_set(): STOP.set()                            # second Ctrl+C: stop right now
        QUIT.set()
    signal.signal(signal.SIGINT, on_ctrl_c)
    t_begin, last_clip, n, jobs = time.time(), -1e9, 0, []
    last_status = time.time()
    new = not os.path.exists(LOG)
    logf = open(LOG, "a", newline=""); w = csv.writer(logf)
    if new: w.writerow(["file", "made_at", "rec_from", "rec_to", "reason", "chat_rate", "normal_rate", "chat_words"])

    def make(t_spike, reason, rate=0.0, base=0.0):
        nonlocal n
        n += 1; name = time.strftime("clip_%Y%m%d_%H%M%S")
        t0, t1 = max(0, t_spike - a.pre), t_spike + a.post
        words = top_words(rec_start + t0, rec_start + t1)
        say(f"{reason}: clipping {t0:.0f}-{t1:.0f}s (chat {rate:.1f}/s vs normal {base:.1f}/s: {words})")

        def job():
            path = cut(t0, t1, name)
            if path:
                say("saved", path); w.writerow([os.path.basename(path), time.strftime("%H:%M:%S"), f"{t0:.1f}", f"{t1:.1f}",
                                                 reason, f"{rate:.2f}", f"{base:.2f}", words]); logf.flush()
        th = threading.Thread(target=job); th.start(); jobs.append(th)

    while not STOP.is_set() and not QUIT.is_set():
        time.sleep(1)
        if rec_start is None: continue
        now = time.time()
        if manual.is_set():
            manual.clear(); make(rec_now() - 2, "manual")
        sec = per_second(now, 160)
        warm = now - t_begin >= a.warmup and len(chat_times) >= 20
        rate = sum(sec[-8:]) / 8                                   # chat comes in batches: use 8-second windows
        hist = sec[:-8][-int(min(152, max(8, now - t_begin - 8))):]
        bins = [sum(hist[i:i + 8]) / 8 for i in range(0, len(hist) - 7, 4)]
        base = max(statistics.median(bins) if bins else 0, 0.2)
        if now - last_status >= 60:
            last_status = now
            m, s_ = divmod(int(rec_now()), 60)
            if not chat_times:
                say(f"status: {m}:{s_:02d} recorded · no chat messages yet (is the chat on?)")
            elif not warm:
                say(f"status: {m}:{s_:02d} recorded · chat {rate:.1f}/s · still learning the normal chat speed")
            else:
                say(f"status: {m}:{s_:02d} recorded · chat {rate:.1f}/s, normal {base:.1f}/s, "
                    f"clips at {max(a.spike * base, a.min_rate):.1f}/s · {n} clip(s)")
        if not warm: continue
        if rate >= max(a.spike * base, a.min_rate) and rec_now() - last_clip > a.cooldown:
            last_clip = rec_now()
            # wait for the reaction to peak (the moment itself is ~5-20 s before the chat peaks)
            make(rec_now(), "chat spike", rate, base)
        if a.max_clips and n >= a.max_clips: break
        if max_dur and now - t_begin > max_dur:
            say(f"Reached stream max time ({max_dur / 3600:.1f}h). Stopping.")
            break
    QUIT.set()
    if any(th.is_alive() for th in jobs):
        say("stopping, finishing clips… (still recording the last seconds; Ctrl+C again to quit now)")
    for th in jobs: th.join()
    STOP.set()
    rec_thread.join(5)                                          # let streamlink/ffmpeg exit before the cleanup
    logf.close()
    if not a.keep: shutil.rmtree(REC, ignore_errors=True)
    say(f"done: {n} clip(s) in {OUT}")


if __name__ == "__main__":
    main()
