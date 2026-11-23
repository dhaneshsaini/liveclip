"""liveclip app: paste a video link (or drop a file), mark the facecam and the gameplay, get vertical clips with captions.

Runs on your own laptop: python app.py  ->  http://127.0.0.1:7860 (the page is static/index.html)
Finished videos have no live chat, so moments are found from loudness peaks (shouting, hype, big reactions).
For live streams use liveclip.py (it clips from chat spikes while the stream is on).
"""
import os, shutil, subprocess, sys, tempfile, threading, time, wave, webbrowser
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "clips")
WORK = os.path.join(tempfile.gettempdir(), "liveclip-app")
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
W, H = 1080, 1920
STYLES = ["Bold white", "Yellow pop", "White box"]
HINGLISH = ("Nahi yaar, yaha se rasta hi nahi hai. Bhai yeh dekho, agar aap naye ho toh follow kar lo. "
            "Main abhi helicopter dhoondh raha hoon, kya scene hai, chalo chalo. Haan bhai, mast hai.")


def run(*cmd):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        err_lines = (r.stderr or "").strip().splitlines()
        raise RuntimeError(err_lines[-1] if err_lines else "command failed")


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def parse_time_str(val):
    """Parse time string like '01:30:00', '45:00', '90', '2h', '1h30m', '120m' into float seconds."""
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
    import re
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*([hms])', val.lower())
    if matches:
        mults = {'h': 3600, 'm': 60, 's': 1}
        return max(0.0, sum(float(n) * mults[u] for n, u in matches))
    return 0.0


def format_time_hms(sec):
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def resolve_time_range(start_val, end_val="", max_val="", total_dur=None):
    """
    Returns (t_start, t_end, t_dur) in seconds.
    Supports:
      - start="02:20:22", end="05:20:12" -> (8422, 19212, 10790)
      - start="00:10:00", end="30m"      -> (600, 2400, 1800)
      - start="00:00:00", end="24:00:00" -> (0, 86400, 86400)
    """
    t_start = parse_time_str(start_val)
    raw_target = str(end_val or "").strip() or str(max_val or "").strip()

    if not raw_target:
        t_end = float(total_dur) if (total_dur and float(total_dur) > t_start) else (t_start + 86400.0)
    else:
        t_parsed = parse_time_str(raw_target)
        if t_parsed <= 0:
            t_end = float(total_dur) if (total_dur and float(total_dur) > t_start) else (t_start + 86400.0)
        elif t_parsed > t_start:
            # User specified an explicit end timestamp (e.g. 05:20:12 > 02:20:22)
            t_end = min(t_parsed, t_start + 86400.0)
        else:
            # User specified a duration (e.g. 30m, 2h)
            t_end = t_start + min(t_parsed, 86400.0)

    t_dur = max(0.0, t_end - t_start)
    return t_start, t_end, t_dur


# ---------- 3. find the loudest moments ----------
def find_moments(src, n, length, t_start=0.0, t_end=None):
    wav = os.path.join(WORK, "all.wav")
    sec = 0
    x = None
    try:
        run("ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", wav)
        if os.path.exists(wav) and os.path.getsize(wav) > 100:
            with wave.open(wav) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
            sec = len(x) // 8000
    except Exception as e:
        print(f"find_moments audio extraction note: {e}", flush=True)

    if sec <= 0:
        sec = int(duration(src))

    win_start = max(0.0, float(t_start or 0.0))
    win_end = min(float(sec), float(t_end)) if (t_end is not None and float(t_end) > win_start) else float(sec)
    win_len = win_end - win_start

    if win_len < length:
        return [win_start]

    picks = []
    if x is not None and len(x) >= 8000 * length:
        s0 = int(win_start * 8000)
        s1 = int(win_end * 8000)
        sub_x = x[s0:s1]
        sub_sec = len(sub_x) // 8000
        if sub_sec >= length:
            rms = np.sqrt((sub_x[:sub_sec * 8000].reshape(sub_sec, 8000) ** 2).mean(1)) + 1
            db = 20 * np.log10(rms)
            score = np.convolve(db - np.median(db), np.ones(6) / 6, "same")
            for i in np.argsort(score)[::-1]:
                rel_start = max(0, min(sub_sec - length, i - int(length * 0.6)))
                abs_start = round(win_start + rel_start, 2)
                if all(abs(abs_start - p) >= length for p in picks):
                    picks.append(abs_start)
                if len(picks) >= n:
                    break

    # If loudness peak detection found fewer than n clips, space them out across [win_start, win_end]
    if len(picks) < n:
        max_start = max(0, win_len - length)
        if max_start > 0:
            step = max_start / max(1, n)
            for k in range(n):
                cand = round(win_start + min(max_start, k * step), 2)
                if all(abs(cand - p) >= (length * 0.4) for p in picks):
                    picks.append(cand)
                if len(picks) >= n:
                    break

    if not picks:
        picks = [win_start]

    return sorted(float(p) for p in picks)


# ---------- 4. captions ----------
_model, _lock = None, threading.Lock()


def whisper():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def load_wav(path):
    with wave.open(path, "rb") as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def captions(wav, lang):
    kw = dict(word_timestamps=True, vad_filter=True, condition_on_previous_text=False)
    if lang == "Hinglish": kw.update(language="en", initial_prompt=HINGLISH)
    elif lang == "English (translate)": kw.update(task="translate")
    samples = load_wav(wav)
    segs, _ = whisper().transcribe(samples, **kw)
    words = [w for s in segs for w in (s.words or [])]
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= 3 or w.start - cur[-1].end > 0.6): out.append(cur); cur = []
        cur.append(w)
    if cur: out.append(cur)
    return [(c[0].start, c[-1].end, " ".join(x.word.strip() for x in c).strip(" ,.")) for c in out]


def font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Black.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "C:/Windows/Fonts/ariblk.ttf", "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f): return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def caption_png(text, style, path):
    up = style != "White box"
    text = text.upper() if up else text
    f = font(76 if style == "Yellow pop" else 64)
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d0.textlength(t, font=f) > W - 180 and cur: lines.append(cur); cur = w
        else: cur = t
    if cur: lines.append(cur)
    lh = int(f.size * 1.25)
    im = Image.new("RGBA", (W, lh * len(lines) + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f); x = (W - tw) / 2; y = 20 + i * lh
        if style == "White box":
            d.rounded_rectangle((x - 24, y - 8, x + tw + 24, y + lh - 4), radius=18, fill=(255, 255, 255, 255))
            d.text((x, y), ln, font=f, fill=(0, 0, 0, 255))
        else:
            col = (255, 225, 77, 255) if style == "Yellow pop" else (255, 255, 255, 255)
            d.text((x, y), ln, font=f, fill=col, stroke_width=8, stroke_fill=(0, 0, 0, 255))
    im.save(path)
    return im.height


# ---------- 5. cut one vertical clip ----------
def make_clip(st, t0, length, face_pct, style, lang, name):
    src, tmp = st["src"], os.path.join(WORK, name)
    os.makedirs(tmp, exist_ok=True)
    face, game = st["boxes"]["face"], st["boxes"]["game"]
    pw, ph = Image.open(os.path.join(WORK, "preview.jpg")).size

    def crop(b):
        x0, y0, x1, y1 = b
        x_min, x_max = max(0, min(x0, x1)), min(pw, max(x0, x1))
        y_min, y_max = max(0, min(y0, y1)), min(ph, max(y0, y1))
        cw = max(2, min(pw, x_max - x_min)) // 2 * 2
        ch = max(2, min(ph, y_max - y_min)) // 2 * 2
        cx = max(0, min(pw - cw, x_min)) // 2 * 2
        cy = max(0, min(ph - ch, y_min)) // 2 * 2
        return f"crop={cw}:{ch}:{cx}:{cy}"

    if face and game:
        fh = int(H * face_pct / 100) // 2 * 2
        vf = (f"[0:v]split[a][b];[a]{crop(face)},scale={W}:{fh}:force_original_aspect_ratio=increase,crop={W}:{fh}[f];"
              f"[b]{crop(game)},scale={W}:{H - fh}:force_original_aspect_ratio=increase,crop={W}:{H - fh}[g];[f][g]vstack[v0]")
        cap_y = fh
    elif face and not game:
        fh = int(H * face_pct / 100) // 2 * 2
        vf = (f"[0:v]split[a][b];[a]{crop(face)},scale={W}:{fh}:force_original_aspect_ratio=increase,crop={W}:{fh}[f];"
              f"[b]scale={W}:{H - fh}:force_original_aspect_ratio=increase,crop={W}:{H - fh}[g];[f][g]vstack[v0]")
        cap_y = fh
    elif game and not face:
        vf = (f"[0:v]{crop(game)},split[a][b];[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,"
              f"scale={W}:{H}[bg];[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
        fh, cap_y = 0, int(H * 0.68)
    else:  # no boxes: whole frame over a blurred copy
        vf = (f"[0:v]split[a][b];[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,"
              f"scale={W}:{H}[bg];[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
        fh, cap_y = 0, int(H * 0.68)

    inputs, last = ["-ss", f"{t0:.2f}", "-t", f"{length:.2f}", "-i", src], "v0"
    if lang != "Off":
        wav = os.path.join(tmp, "a.wav")
        caps = []
        try:
            run("ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.2f}", "-t", f"{length:.2f}", "-i", src, "-vn", "-ac", "1",
                "-ar", "16000", "-c:a", "pcm_s16le", wav)
            caps = captions(wav, lang)
        except Exception as e:
            print(f"Captions skipped for {name}: {e}", flush=True)

        for i, (s, e, txt) in enumerate(caps):
            png = os.path.join(tmp, f"c{i}.png")
            h = caption_png(txt, style, png)
            inputs += ["-i", png]
            vf += f";[{last}][{i + 1}:v]overlay=0:{max(0, cap_y - h // 2)}:enable='between(t,{s:.2f},{e + 0.1:.2f})'[v{i + 1}]"
            last = f"v{i + 1}"

    out = os.path.join(OUT, name + ".mp4")
    run("ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", vf, "-map", f"[{last}]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out)
    return out



# ---------- web API ----------
app = FastAPI()
ST = {"src": None, "dur": 0, "start_time": 0.0, "end_time": 0.0, "max_time": 0.0, "boxes": {"face": None, "game": None}}
JOB = {"busy": False, "progress": 0.0, "msg": "", "clips": [], "error": ""}


def preview_frame():
    prev = os.path.join(WORK, "preview.jpg")
    frame_pos = max(0.5, ST["dur"] * 0.3)
    run("ffmpeg", "-v", "error", "-y", "-ss", f"{frame_pos:.1f}", "-i", ST["src"], "-frames:v", "1", prev)
    w, h = Image.open(prev).size
    return {
        "preview": f"/api/preview.jpg?t={time.time():.0f}",
        "w": w,
        "h": h,
        "dur": round(ST["dur"], 2),
        "dur_hms": format_time_hms(ST["dur"]),
        "start_time": ST.get("start_time", 0.0),
        "end_time": ST.get("end_time", 0.0),
        "start_hms": format_time_hms(ST.get("start_time", 0.0)),
        "end_hms": format_time_hms(ST.get("end_time", 0.0)),
    }


@app.post("/api/load")
async def api_load(
    url: str = Form(""),
    file: UploadFile | None = File(None),
    start_time: str = Form(""),
    end_time: str = Form(""),
    max_time: str = Form(""),
):
    if JOB["busy"]: raise HTTPException(409, "Still making clips, wait for them to finish.")
    shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
    src = os.path.join(WORK, "src.mp4")
    t_start, t_end, t_dur = resolve_time_range(start_time, end_time, max_time)

    if file is not None and file.filename:
        raw = os.path.join(WORK, "raw_" + file.filename)
        with open(raw, "wb") as f: shutil.copyfileobj(file.file, f)
        raw_dur = duration(raw)
        t_start, t_end, t_dur = resolve_time_range(start_time, end_time, max_time, total_dur=raw_dur)
        if t_start > 0 or (t_end > 0 and t_end < (raw_dur - 0.5)):
            clip_secs = max(1.0, t_end - t_start)
            try:
                run("ffmpeg", "-v", "error", "-y", "-ss", f"{t_start:.2f}", "-t", f"{clip_secs:.2f}", "-i", raw, "-c", "copy", src)
                if not os.path.exists(src) or duration(src) < 0.5:
                    raise RuntimeError("copy trim too short")
            except Exception:
                run("ffmpeg", "-v", "error", "-y", "-ss", f"{t_start:.2f}", "-t", f"{clip_secs:.2f}",
                    "-i", raw, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", src)
        else:
            shutil.move(raw, src)
    elif url.strip():
        u = url.strip()
        cmd = [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
               "--merge-output-format", "mp4"]
        download_section = (t_start > 0 or (t_end > 0 and t_end < 86400))
        if download_section:
            end_sec = min(t_end, t_start + 86400.0)
            cmd += ["--download-sections", f"*{format_time_hms(t_start)}-{format_time_hms(end_sec)}",
                    "--force-keyframes-at-cuts"]
        cmd += ["-o", src, "--no-playlist", u]
        try:
            run(*cmd)
        except Exception as e:
            if download_section:
                cmd_fb = [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
                          "--merge-output-format", "mp4", "-o", src, "--no-playlist", u]
                try:
                    run(*cmd_fb)
                except Exception as e2:
                    raise HTTPException(400, f"Couldn't download that link ({e2}). Check the URL.")
            else:
                raise HTTPException(400, f"Couldn't download that link. Check it opens in your browser. ({e})")
        if not os.path.exists(src):
            candidates = [os.path.join(WORK, f) for f in os.listdir(WORK) if f.endswith((".mp4", ".mkv", ".webm", ".ts"))]
            if candidates:
                shutil.move(candidates[0], src)
            else:
                raise HTTPException(400, "Downloaded video file could not be found.")
    else:
        raise HTTPException(400, "Paste a link or choose a video file first.")

    total_src_dur = duration(src)
    ST.update(src=src, dur=total_src_dur, start_time=t_start, end_time=t_end, max_time=t_dur, boxes={"face": None, "game": None})
    return preview_frame()


@app.get("/api/preview.jpg")
def api_preview():
    return FileResponse(os.path.join(WORK, "preview.jpg"))


def worker(opts):
    try:
        JOB.update(busy=True, progress=0.03, msg="Finding the best moments", clips=[], error="")
        n_clips = max(1, min(1000, int(opts.get("n", 4))))
        clip_len = max(5, int(opts.get("length", 30)))
        face_pct = int(opts.get("face_pct", 35))
        style = opts.get("style", "Yellow pop")
        lang = opts.get("lang", "Hinglish")

        # Time range selection for clipping
        clip_start = opts.get("start_time", "")
        clip_end = opts.get("end_time", "") or opts.get("max_time", "")
        if clip_start or clip_end:
            t_start, t_end, _ = resolve_time_range(clip_start, clip_end, total_dur=ST["dur"])
        else:
            t_start = 0.0
            t_end = ST["dur"]

        starts = find_moments(ST["src"], n_clips, clip_len, t_start=t_start, t_end=t_end)
        if not starts:
            starts = [t_start]
        base = "clip_" + time.strftime("%Y%m%d_%H%M%S")
        for k, t0 in enumerate(starts):
            JOB.update(progress=0.08 + 0.9 * k / max(1, len(starts)), msg=f"Making clip {k + 1} of {len(starts)}")
            out = make_clip(ST, t0, clip_len, face_pct, style, lang, f"{base}_{int(t0)}s")
            JOB["clips"].append("/clips/" + os.path.basename(out))
        JOB.update(progress=1.0, msg=f"Done: {len(starts)} clips")
    except Exception as e:
        import traceback
        traceback.print_exc()
        JOB.update(error=str(e), msg=f"Error: {e}")
    finally:
        JOB["busy"] = False


@app.post("/api/clip")
def api_clip(opts: dict):
    if not ST["src"]: raise HTTPException(400, "Load a video first.")
    if JOB["busy"]: raise HTTPException(409, "Already making clips.")
    def box(b):
        return tuple(int(v) for v in b) if b and abs(b[2] - b[0]) > 8 and abs(b[3] - b[1]) > 8 else None
    ST["boxes"] = {"face": box(opts.get("face")), "game": box(opts.get("game"))}
    JOB.update(busy=True, progress=0.0, msg="Starting", clips=[], error="")
    threading.Thread(target=worker, args=(opts,), daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def api_status():
    return JOB


@app.post("/api/open-folder")
def api_open_folder():
    subprocess.Popen(["open" if sys.platform == "darwin" else "explorer" if os.name == "nt" else "xdg-open", OUT])
    return {"ok": True}


app.mount("/clips", StaticFiles(directory=OUT), name="clips")
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"liveclip is running at http://127.0.0.1:{port}  (keep this window open)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
