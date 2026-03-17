"""
Gemini Batch API — Junk Journal Paper Image Generator
=====================================================
Generates diverse junk journal / scrapbook paper designs using the
Gemini Batch API (gemini-3-pro-image-preview) at 50 % batch pricing.

Usage:
    1. Copy .env.example → .env  and fill in your GEMINI_API_KEY
    2. pip install -r requirements.txt
    3. python batch_generate.py
"""

import json
import os
import sys
import time
import base64
import pathlib

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("ERROR: Set GEMINI_API_KEY in your .env file (see .env.example)")

MODEL = "gemini-3-pro-image-preview"
JSONL_FILE = "batch-requests.jsonl"
OUTPUT_DIR = pathlib.Path("output")
POLL_INTERVAL = 30  # seconds between status checks

# ── Prompt Library ───────────────────────────────────────────────────────────
# Each prompt asks for a FULL-PAGE seamless paper texture / background.
# No text on the image — pure visual pattern.

PROMPTS = [
    # ── Vintage & Aged ────────────────────────────────────────────────────
    "A full-page seamless junk journal paper background: aged antique parchment with warm tea-stained tones, subtle foxing spots, slightly crinkled edges, and a faint yellowed patina. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: old vintage ledger paper with faded blue and red ruling lines on cream paper, showing authentic aging and light water stains. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: distressed vintage book page texture with yellowed ivory color, coffee-ring stains, soft creases, and worn fibrous texture. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: antique love-letter paper with delicate faded rose-tinted edges, soft sepia watercolor washes, and a romantic vintage feel. No text, no letters, no words.",

    # ── Floral & Botanical ────────────────────────────────────────────────
    "A full-page seamless junk journal paper background: delicate botanical illustration style with soft watercolor wildflowers — poppies, lavender, daisies — scattered on aged cream paper. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: vintage pressed flower arrangement with dried roses, ferns, and baby's breath on warm parchment, victorian herbarium style. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: English cottage garden floral pattern in muted dusty pink, sage green, and antique gold on a soft ivory background. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: dark moody floral design with deep burgundy roses, dark foliage, and gold accents on a nearly black background, Dutch masters painting style. No text, no letters, no words.",

    # ── Maps & Travel ─────────────────────────────────────────────────────
    "A full-page seamless junk journal paper background: vintage world map fragment with aged sepia tones, compass roses, dotted travel routes, and antique cartography styling. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: old travel journal texture with postage-stamp collage elements, faded passport stamps, vintage airmail stripes, and cream paper. No text, no letters, no words.",

    # ── Music & Art ───────────────────────────────────────────────────────
    "A full-page seamless junk journal paper background: vintage handwritten music sheet on aged ivory paper with faded brown staff lines and musical notes, showing authentic aging and staining. No text, no letters, no words — only musical notation.",

    "A full-page seamless junk journal paper background: watercolor artist palette texture with soft blended washes of dusty rose, sage, lavender, and gold on thick textured watercolor paper. No text, no letters, no words.",

    # ── Lace & Fabric ─────────────────────────────────────────────────────
    "A full-page seamless junk journal paper background: delicate white lace doily pattern overlay on soft blush pink aged paper, with intricate crochet-style lacework details. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: vintage burlap and linen fabric texture in natural tan and cream tones, showing woven fiber detail and a rustic handmade feel. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: french toile de jouy pattern in faded blue ink on antique cream linen, showing pastoral countryside scenes in a repeating pattern. No text, no letters, no words.",

    # ── Steampunk & Industrial ────────────────────────────────────────────
    "A full-page seamless junk journal paper background: steampunk style with brass gears, cog wheels, copper rivets, and mechanical sketches on aged brown kraft paper. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: industrial blueprint style with faded white technical drawings of gears and mechanisms on deep prussian blue paper, vintage engineering aesthetic. No text, no letters, no words.",

    # ── Watercolor & Abstract ─────────────────────────────────────────────
    "A full-page seamless junk journal paper background: abstract watercolor wash in ocean tones — soft teal, deep navy, seafoam green — on textured cold-press watercolor paper with natural paint bleeding. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: galaxy-inspired watercolor with deep indigo, purple, and touches of gold splatter on black paper, creating a magical celestial night sky effect. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: autumn harvest watercolor in warm burnt sienna, golden amber, and deep rust tones, with soft blended leaf silhouettes on cream paper. No text, no letters, no words.",

    # ── Collage & Ephemera ────────────────────────────────────────────────
    "A full-page seamless junk journal paper background: vintage ephemera collage with layered old tickets, postage stamps, receipts, tags, and labels in sepia and muted rose tones, artfully overlapping on cream paper. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: victorian scrapbook style with layered die-cut scraps of cherubs, birds, flowers, and ribbons on aged cream cardstock. No text, no letters, no words.",

    # ── Nature & Organic ──────────────────────────────────────────────────
    "A full-page seamless junk journal paper background: pressed autumn leaves — maple, oak, and birch — in warm red, orange, and gold on handmade recycled paper with visible fiber inclusions. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: japanese washi paper texture in soft sage green with delicate gold leaf flecks and visible kozo fibers, zen minimalist aesthetic. No text, no letters, no words.",

    # ── Grunge & Distressed ───────────────────────────────────────────────
    "A full-page seamless junk journal paper background: heavily distressed grunge texture in dark charcoal and warm brown tones, layered paint peels, scratches, and urban decay aesthetic on thick paper. No text, no letters, no words.",

    "A full-page seamless junk journal paper background: vintage newspaper texture with heavily faded and yellowed newsprint, ink bleed, and water damage creating an abstract pattern on fragile paper. No text, no letters, no words.",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_jsonl(prompts: list[str], path: str) -> str:
    """Write prompts into a JSONL file formatted for the Gemini Batch API."""
    with open(path, "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts, start=1):
            entry = {
                "key": f"junk-journal-{idx:03d}",
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generation_config": {
                        "responseModalities": ["TEXT", "IMAGE"],
                    },
                },
            }
            f.write(json.dumps(entry) + "\n")
    print(f"✅  Wrote {len(prompts)} requests → {path}")
    return path


def upload_file(client: genai.Client, path: str) -> str:
    """Upload JSONL to the Gemini File API and return the file resource name."""
    uploaded = client.files.upload(
        file=path,
        config=types.UploadFileConfig(
            display_name="junk-journal-batch-requests",
            mime_type="jsonl",
        ),
    )
    print(f"✅  Uploaded file: {uploaded.name}")
    return uploaded.name


def create_batch_job(client: genai.Client, file_name: str):
    """Create a batch job pointing at the uploaded JSONL file."""
    job = client.batches.create(
        model=MODEL,
        src=file_name,
        config={"display_name": "junk-journal-paper-batch"},
    )
    print(f"✅  Batch job created: {job.name}")
    return job


COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def poll_until_done(client: genai.Client, job_name: str):
    """Poll the batch job until it reaches a terminal state."""
    print(f"\n⏳  Polling job: {job_name}  (every {POLL_INTERVAL}s)")
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        print(f"    → {state}")
        if state in COMPLETED_STATES:
            return job
        time.sleep(POLL_INTERVAL)


def save_images(client: genai.Client, job):
    """Download result JSONL, extract base64 images, save as PNGs."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    saved = 0

    # ── File-based results ────────────────────────────────────────────────
    if job.dest and job.dest.file_name:
        result_file = job.dest.file_name
        print(f"\n📥  Downloading results from: {result_file}")
        raw = client.files.download(file=result_file)
        lines = raw.decode("utf-8").splitlines()

        for line in lines:
            if not line.strip():
                continue
            parsed = json.loads(line)
            key = parsed.get("key", f"unknown-{saved+1}")

            if "error" in parsed and parsed["error"]:
                print(f"   ⚠️  {key}: {parsed['error']}")
                continue

            response = parsed.get("response")
            if not response:
                continue

            candidates = response.get("candidates", [])
            for candidate in candidates:
                parts = candidate.get("content", {}).get("parts", [])
                for part in parts:
                    if part.get("inlineData"):
                        img_data = base64.b64decode(part["inlineData"]["data"])
                        mime = part["inlineData"].get("mimeType", "image/png")
                        ext = "png" if "png" in mime else "jpg"
                        out_path = OUTPUT_DIR / f"{key}.{ext}"
                        out_path.write_bytes(img_data)
                        saved += 1
                        print(f"   🖼️  Saved: {out_path}")

    # ── Inline results (fallback for small jobs) ──────────────────────────
    elif job.dest and job.dest.inlined_responses:
        print("\n📥  Processing inline responses …")
        for i, resp in enumerate(job.dest.inlined_responses, start=1):
            if resp.error:
                print(f"   ⚠️  Response {i}: {resp.error}")
                continue
            if not resp.response:
                continue
            for candidate in resp.response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        img_data = base64.b64decode(part.inline_data.data)
                        ext = "png" if "png" in part.inline_data.mime_type else "jpg"
                        out_path = OUTPUT_DIR / f"junk-journal-{i:03d}.{ext}"
                        out_path.write_bytes(img_data)
                        saved += 1
                        print(f"   🖼️  Saved: {out_path}")

    print(f"\n🎉  Done! {saved} images saved to {OUTPUT_DIR.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Junk Journal Paper — Gemini Batch Image Generator")
    print("=" * 60)

    client = genai.Client(api_key=API_KEY)

    # 1. Build JSONL
    build_jsonl(PROMPTS, JSONL_FILE)

    # 2. Upload
    file_name = upload_file(client, JSONL_FILE)

    # 3. Create batch job
    job = create_batch_job(client, file_name)

    # 4. Poll
    job = poll_until_done(client, job.name)

    # 5. Handle result
    if job.state.name == "JOB_STATE_SUCCEEDED":
        save_images(client, job)
    else:
        print(f"\n❌  Job ended with state: {job.state.name}")
        if hasattr(job, "error") and job.error:
            print(f"    Error: {job.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
