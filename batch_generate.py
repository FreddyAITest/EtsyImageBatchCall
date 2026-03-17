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
# Professional printable digital paper prompts for Etsy junk journal products.
# Key rules: edge-to-edge coverage, no torn/ragged paper edges, digital art
# quality, rich decorative design, portrait orientation implied.

# Common suffix appended to every prompt for consistency
_SUFFIX = (
    " The design must fill the entire image edge-to-edge with no visible paper "
    "borders, no torn edges, no ragged edges, and no white margins. "
    "Printable digital paper product, high resolution, professional quality. "
    "Do not include any readable text, letters, words, or watermarks."
)

PROMPTS = [
    # ── Shabby Chic Vintage ───────────────────────────────────────────────
    "Printable digital scrapbook paper design: shabby chic vintage style with layered soft pink and cream roses, delicate lace ribbon borders, faded damask pattern in the background, pearl accents, and a warm antique ivory base. Hand-painted digital illustration style." + _SUFFIX,

    "Printable digital scrapbook paper design: romantic vintage collage with layered watercolor rose bouquets, antique clock faces, ornate golden frames, faded sheet music fragments, and butterfly silhouettes on a warm tea-stained cream background. Rich layered digital art." + _SUFFIX,

    "Printable digital scrapbook paper design: vintage apothecary aesthetic with illustrated botanical herbs, antique medicine bottles, dried flower specimens, ornate Victorian filigree borders, and aged sepia-toned background with subtle parchment texture. Detailed digital illustration." + _SUFFIX,

    "Printable digital scrapbook paper design: French provincial style with delicate toile de jouy pastoral scenes in dusty blue on antique cream, overlaid with faded rose garlands, vintage postmark motifs, and a soft watercolor wash effect. Elegant hand-painted digital art." + _SUFFIX,

    # ── Dark & Moody Florals ──────────────────────────────────────────────
    "Printable digital scrapbook paper design: dark moody floral wallpaper with lush deep burgundy garden roses, dark emerald leaves, golden seed pods, and tiny white blooms on a rich charcoal black background. Dutch Golden Age still-life painting style, digitally rendered." + _SUFFIX,

    "Printable digital scrapbook paper design: gothic romantic style with deep plum and wine-colored peonies, dark trailing ivy, antique gold filigree swirls, and midnight blue watercolor undertones. Dramatic and luxurious hand-painted digital art." + _SUFFIX,

    "Printable digital scrapbook paper design: mystical enchanted forest with deep teal background, luminous moonlit mushrooms, delicate ferns, fireflies rendered as gold dots, and trailing wisteria vines. Whimsical fairy-tale digital illustration style." + _SUFFIX,

    # ── Cottagecore & Garden ───────────────────────────────────────────────
    "Printable digital scrapbook paper design: cottagecore aesthetic with a meadow of hand-painted wildflowers — daisies, cornflowers, poppies, and Queen Anne's lace — on a soft sage green watercolor wash background. Fresh and dreamy digital illustration." + _SUFFIX,

    "Printable digital scrapbook paper design: English cottage garden repeating pattern with climbing roses, lavender sprigs, sweet peas, and garden birds in soft muted pink, lilac, sage, and butter yellow on warm cream. Charming watercolor illustration style." + _SUFFIX,

    "Printable digital scrapbook paper design: vintage seed packet collage art with illustrated vegetables, flowers, garden tools, and ornate Victorian typography frames arranged in a decorative repeating layout on warm kraft-colored background. Retro botanical illustration style." + _SUFFIX,

    # ── Watercolor Washes ─────────────────────────────────────────────────
    "Printable digital scrapbook paper design: dreamy watercolor marble texture in soft blush pink, rose gold, and ivory white with flowing organic veins and subtle gold foil splatter accents. Elegant abstract modern digital art." + _SUFFIX,

    "Printable digital scrapbook paper design: ocean-inspired watercolor wash with flowing layers of deep teal, cerulean blue, seafoam green, and white foam patterns, accented with tiny gold foil starfish and shells scattered throughout. Artistic hand-painted digital design." + _SUFFIX,

    "Printable digital scrapbook paper design: celestial galaxy watercolor with deep indigo, rich purple, and midnight blue swirls, scattered with gold leaf constellation patterns, crescent moons, and twinkling stars. Magical and luxurious digital art." + _SUFFIX,

    "Printable digital scrapbook paper design: autumn harvest watercolor with rich layers of burnt sienna, amber gold, deep rust, and warm chocolate brown flowing together, overlaid with delicate gilded leaf silhouettes — maple, oak, and birch. Warm organic digital art." + _SUFFIX,

    # ── Vintage Travel & Ephemera ─────────────────────────────────────────
    "Printable digital scrapbook paper design: vintage travel ephemera collage with layered antique world map fragments, ornate compass roses, illustrated hot air balloons, old postage stamps, wax seal motifs, and faded route lines on warm sepia-toned background. Rich layered digital collage art." + _SUFFIX,

    "Printable digital scrapbook paper design: Parisian vintage aesthetic with illustrated Eiffel Tower vignettes, ornate iron scrollwork patterns, French postcard motifs, delicate rose garlands, and vintage perfume bottle sketches on a soft blush-and-gold background. Elegant digital illustration." + _SUFFIX,

    # ── Steampunk & Mechanical ────────────────────────────────────────────
    "Printable digital scrapbook paper design: steampunk aesthetic with intricately detailed brass clockwork gears, copper mechanical parts, ornate Victorian key illustrations, airship sketches, and industrial riveted metal panel textures on a rich dark brown background. Detailed technical illustration style." + _SUFFIX,

    "Printable digital scrapbook paper design: vintage engineering blueprint aesthetic with detailed mechanical patent drawings, gear schematics, and ornate technical diagrams rendered in cream lines on deep Prussian blue background, with subtle aged paper texture overlay. Technical illustration style." + _SUFFIX,

    # ── Lace & Textile ────────────────────────────────────────────────────
    "Printable digital scrapbook paper design: layered antique lace design with intricate Chantilly lace patterns in ivory white overlaid on a soft dusty rose background, with scattered pearl bead accents and satin ribbon bow details. Delicate and feminine digital art." + _SUFFIX,

    "Printable digital scrapbook paper design: vintage patchwork quilt pattern with assorted fabric swatches — tiny florals, gingham checks, polka dots, and calico prints — stitched together with visible cross-stitch borders in warm farmhouse colors. Cozy hand-crafted digital illustration." + _SUFFIX,

    # ── Victorian & Art Nouveau ───────────────────────────────────────────
    "Printable digital scrapbook paper design: Art Nouveau style with flowing organic lines, stylized peacock feather motifs, ornate curving borders, and jewel-toned colors — emerald green, sapphire blue, and antique gold — on a rich cream background. Elegant decorative digital art." + _SUFFIX,

    "Printable digital scrapbook paper design: Victorian Christmas aesthetic with illustrated poinsettias, holly boughs with red berries, golden bells, ornate frames, vintage Santa vignettes, and snowflake accents on a rich forest green background with gold foil details. Festive digital illustration." + _SUFFIX,

    # ── Botanical & Nature ────────────────────────────────────────────────
    "Printable digital scrapbook paper design: detailed botanical specimen illustration with precisely rendered pressed flowers, ferns, and leaves in the style of a Victorian herbarium plate, with handwritten-style labels (non-readable decorative script), on aged warm ivory background. Scientific illustration style." + _SUFFIX,

    "Printable digital scrapbook paper design: Japanese-inspired washi paper design with delicate cherry blossom branches, koi fish, tiny origami cranes, and gold leaf cloud motifs on a soft sage green watercolor background. Zen minimalist asian-inspired digital art." + _SUFFIX,

    "Printable digital scrapbook paper design: enchanted woodland scene with illustrated acorns, pinecones, fern fronds, tiny woodland mushrooms, and forest berries scattered across a soft mossy green and warm brown watercolor background. Charming nature-inspired digital illustration." + _SUFFIX,

    # ── Grunge & Mixed Media ──────────────────────────────────────────────
    "Printable digital scrapbook paper design: mixed media art journal style with layered paint textures, stamped geometric patterns, ink splatter, washi tape strips, and torn paper collage elements in a curated palette of mustard yellow, teal, coral, and charcoal. Artistic and textured digital design." + _SUFFIX,

    "Printable digital scrapbook paper design: vintage music and art collage with layered antique sheet music notation, watercolor paint swatches, illustrated musical instruments, ink blots, and ornate border frames on a warm coffee-toned background. Artistic layered digital collage." + _SUFFIX,
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
