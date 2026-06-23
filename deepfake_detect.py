"""
deepfake_detect.py
==================
Deepfake & Image Forgery Detection — Unified Pipeline
Australian Phoenix CyberOps | Chadi Saliby

Runs three forensic detection methods against any JPEG image:
  1. EXIF Metadata Audit
  2. Error Level Analysis (ELA)
  3. Discrete Fourier Transform (DFT) Frequency Analysis

Usage:
    python deepfake_detect.py --image Lab/suspect.jpg

Output:
    - Console: EXIF audit results with flagged fields
    - Image:   Lab/suspect_analysis.png  (3-panel forensic report)
    - Report:  Lab/suspect_report.txt    (plain-text findings summary)

Requirements:
    pip install Pillow numpy matplotlib exifread
"""

import argparse
import io
import os
import sys
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageChops, ImageEnhance

try:
    import exifread
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False
    print("[!] exifread not installed — EXIF audit will be skipped.")
    print("    Run: pip install exifread\n")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EDITING_SOFTWARE = [
    "photoshop", "gimp", "affinity", "lightroom",
    "pixelmator", "canva", "snapseed", "facetune",
    "dall-e", "midjourney", "stable diffusion",
]

EXIF_KEY_FIELDS = [
    "Image DateTime",
    "Image Make",
    "Image Model",
    "EXIF DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "Image Software",
    "GPS GPSLatitude",
    "GPS GPSLongitude",
    "Image ImageDescription",
    "Thumbnail Compression",
]

ELA_QUALITY   = 90    # JPEG re-save quality for ELA comparison
ELA_SCALE     = 15    # Brightness amplification factor for ELA visualisation
DFT_THRESHOLD = 40.0  # dB above median to flag a spectral spike as anomalous


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — EXIF METADATA AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def audit_exif(path):
    """
    Reads EXIF metadata and returns a dict of findings.
    Flags: editing software, timestamp mismatches, missing camera fields.
    """
    findings = {
        "fields":   {},
        "flags":    [],
        "verdict":  "PASS",
    }

    if not EXIF_AVAILABLE:
        findings["flags"].append("exifread not installed — EXIF audit skipped")
        findings["verdict"] = "SKIPPED"
        return findings

    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    if not tags:
        findings["flags"].append("No EXIF metadata found — image may be stripped")
        findings["verdict"] = "FLAG"
        return findings

    # Collect key fields
    for field in EXIF_KEY_FIELDS:
        val = tags.get(field)
        findings["fields"][field] = str(val) if val else "NOT FOUND"

    # ── Flag: editing software ────────────────────────────────────────────────
    software = str(tags.get("Image Software", "")).lower()
    if software:
        for sw in EDITING_SOFTWARE:
            if sw in software:
                findings["flags"].append(
                    f"Editing software detected in metadata: {tags.get('Image Software')}"
                )
                findings["verdict"] = "FLAG"
                break

    # ── Flag: timestamp mismatch ─────────────────────────────────────────────
    create  = str(tags.get("Image DateTime", ""))
    capture = str(tags.get("EXIF DateTimeOriginal", ""))
    if create and capture and create != capture:
        findings["flags"].append(
            f"Timestamp mismatch — DateTime: {create}  |  DateTimeOriginal: {capture}"
        )
        findings["verdict"] = "FLAG"

    # ── Flag: missing camera make / model ────────────────────────────────────
    if not tags.get("Image Make") and not tags.get("Image Model"):
        findings["flags"].append(
            "No camera make or model — consistent with synthetic or stripped image"
        )
        if findings["verdict"] == "PASS":
            findings["verdict"] = "WARN"

    # ── Flag: GPS present but camera absent ──────────────────────────────────
    has_gps    = bool(tags.get("GPS GPSLatitude"))
    has_camera = bool(tags.get("Image Make") or tags.get("Image Model"))
    if has_gps and not has_camera:
        findings["flags"].append(
            "GPS coordinates present but no camera metadata — unusual combination"
        )
        findings["verdict"] = "FLAG"

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ERROR LEVEL ANALYSIS (ELA)
# ─────────────────────────────────────────────────────────────────────────────

def ela(path, quality=ELA_QUALITY, scale=ELA_SCALE):
    """
    Re-saves the image at a known JPEG quality, computes the pixel-wise
    difference, and amplifies it. Returns the ELA image and a finding dict.

    Authentic images: low, uniform brightness across the ELA map.
    Composited regions: anomalously bright patches at manipulation boundaries.
    """
    original = Image.open(path).convert("RGB")

    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    diff    = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    max_val = max(extrema[0][1], extrema[1][1], extrema[2][1])

    factor   = (255.0 / max_val * scale) if max_val > 0 else scale
    ela_img  = ImageEnhance.Brightness(diff).enhance(factor)
    ela_arr  = np.array(ela_img)

    # Statistical summary for the report
    mean_ela = float(ela_arr.mean())
    std_ela  = float(ela_arr.std())
    peak_ela = float(ela_arr.max())

    # Heuristic thresholds (tuned for q=90, scale=15)
    verdict = "PASS"
    flags   = []

    if mean_ela > 18:
        flags.append(f"Elevated mean ELA level ({mean_ela:.1f}) — possible widespread manipulation")
        verdict = "FLAG"
    if std_ela > 22:
        flags.append(f"High ELA variance ({std_ela:.1f}) — inconsistent compression history across image")
        verdict = "FLAG"
    if peak_ela > 200:
        flags.append(f"Extreme ELA peak ({peak_ela:.0f}) — localised region with very different compression history")
        verdict = "FLAG"

    findings = {
        "ela_image": ela_img,
        "mean":      mean_ela,
        "std":       std_ela,
        "peak":      peak_ela,
        "flags":     flags,
        "verdict":   verdict,
    }
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — DFT FREQUENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def dft_analysis(path, spike_threshold_db=DFT_THRESHOLD):
    """
    Applies a 2D Discrete Fourier Transform to the greyscale image and
    inspects the power spectrum for periodic grid artefacts introduced by
    GAN upsampling layers.

    Real images: broadband noise, no visible periodic grid.
    GAN / diffusion output: spectral peaks at fixed stride intervals.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)

    f       = np.fft.fft2(arr)
    fshift  = np.fft.fftshift(f)
    mag     = 20 * np.log(np.abs(fshift) + 1)

    # Remove DC component (centre peak is always dominant in real images)
    h, w = mag.shape
    cx, cy = w // 2, h // 2
    mag_no_dc = mag.copy()
    mag_no_dc[cy-5:cy+5, cx-5:cx+5] = 0

    median_val = float(np.median(mag_no_dc[mag_no_dc > 0]))
    peak_val   = float(mag_no_dc.max())
    peak_above = peak_val - median_val

    verdict = "PASS"
    flags   = []

    if peak_above > spike_threshold_db:
        flags.append(
            f"Spectral spike {peak_above:.1f} dB above median — "
            f"consistent with GAN upsampling artefact (grid periodicity)"
        )
        verdict = "FLAG"

    findings = {
        "magnitude":   mag,
        "median_db":   median_val,
        "peak_db":     peak_val,
        "peak_above":  peak_above,
        "flags":       flags,
        "verdict":     verdict,
    }
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT — 3-PANEL FIGURE
# ─────────────────────────────────────────────────────────────────────────────

VERDICT_COLOUR = {
    "PASS":    "#44BB77",
    "WARN":    "#FFAA33",
    "FLAG":    "#FF4444",
    "SKIPPED": "#888888",
}

def render_figure(path, exif_r, ela_r, dft_r, out_path):
    """Saves a 3-panel forensic figure to out_path."""
    original = Image.open(path).convert("RGB")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="#0D1117")
    fig.suptitle(
        f"Deepfake & Image Forgery Detection Pipeline\n{os.path.basename(path)}",
        fontsize=14, fontweight="bold", color="white", y=0.98
    )

    # ── Panel 1: Original ────────────────────────────────────────────────────
    axes[0].imshow(original)
    axes[0].set_title("① Original Image", color="white", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")
    axes[0].set_facecolor("#0D1117")

    exif_col = VERDICT_COLOUR.get(exif_r["verdict"], "#AAAAAA")
    axes[0].text(
        0.5, -0.04,
        f"EXIF Audit: {exif_r['verdict']}",
        transform=axes[0].transAxes, ha="center", fontsize=10,
        fontweight="bold", color=exif_col
    )
    if exif_r["flags"]:
        summary = exif_r["flags"][0][:60] + ("…" if len(exif_r["flags"][0]) > 60 else "")
        axes[0].text(
            0.5, -0.09, summary,
            transform=axes[0].transAxes, ha="center", fontsize=8,
            color="#CCCCCC", style="italic", wrap=True
        )

    # ── Panel 2: ELA ─────────────────────────────────────────────────────────
    im2 = axes[1].imshow(ela_r["ela_image"], cmap="hot")
    axes[1].set_title(
        "② Error Level Analysis (ELA)\nBright zones = inconsistent compression history",
        color="white", fontsize=10, fontweight="bold", pad=10
    )
    axes[1].axis("off")
    axes[1].set_facecolor("#0D1117")
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.02)
    cb2.ax.tick_params(labelcolor="white", labelsize=8)

    ela_col = VERDICT_COLOUR.get(ela_r["verdict"], "#AAAAAA")
    axes[1].text(
        0.5, -0.04,
        f"ELA: {ela_r['verdict']}  |  mean={ela_r['mean']:.1f}  peak={ela_r['peak']:.0f}",
        transform=axes[1].transAxes, ha="center", fontsize=10,
        fontweight="bold", color=ela_col
    )

    # ── Panel 3: DFT ─────────────────────────────────────────────────────────
    im3 = axes[2].imshow(dft_r["magnitude"], cmap="inferno")
    axes[2].set_title(
        "③ DFT Frequency Spectrum\nGrid spikes = GAN upsampling artefact",
        color="white", fontsize=10, fontweight="bold", pad=10
    )
    axes[2].axis("off")
    axes[2].set_facecolor("#0D1117")
    cb3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.02)
    cb3.ax.tick_params(labelcolor="white", labelsize=8)

    dft_col = VERDICT_COLOUR.get(dft_r["verdict"], "#AAAAAA")
    axes[2].text(
        0.5, -0.04,
        f"DFT: {dft_r['verdict']}  |  spike {dft_r['peak_above']:.1f} dB above median",
        transform=axes[2].transAxes, ha="center", fontsize=10,
        fontweight="bold", color=dft_col
    )

    # ── Overall verdict banner ────────────────────────────────────────────────
    verdicts   = [exif_r["verdict"], ela_r["verdict"], dft_r["verdict"]]
    if "FLAG" in verdicts:
        overall, overall_col = "HIGH PROBABILITY OF MANIPULATION", "#FF4444"
    elif "WARN" in verdicts:
        overall, overall_col = "INDICATORS PRESENT — FURTHER INVESTIGATION RECOMMENDED", "#FFAA33"
    else:
        overall, overall_col = "NO MANIPULATION INDICATORS DETECTED", "#44BB77"

    fig.text(
        0.5, 0.01, f"VERDICT: {overall}",
        ha="center", fontsize=12, fontweight="bold",
        color=overall_col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#111820", edgecolor=overall_col, linewidth=2)
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT — PLAIN-TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(path, exif_r, ela_r, dft_r, out_path):
    """Writes a plain-text forensic findings report."""
    lines = []
    sep   = "=" * 72
    thin  = "-" * 72

    lines.append(sep)
    lines.append("  DEEPFAKE & IMAGE FORGERY DETECTION — FORENSIC REPORT")
    lines.append("  Australian Phoenix CyberOps | deepfake_detect.py")
    lines.append(sep)
    lines.append(f"  File     : {os.path.abspath(path)}")
    lines.append(f"  Size     : {os.path.getsize(path):,} bytes")
    lines.append(f"  Run at   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # Verdicts
    lines.append("SUMMARY VERDICTS")
    lines.append(thin)
    lines.append(f"  EXIF Metadata Audit : {exif_r['verdict']}")
    lines.append(f"  Error Level Analysis: {ela_r['verdict']}")
    lines.append(f"  DFT Frequency Audit : {dft_r['verdict']}")
    lines.append("")

    verdicts = [exif_r["verdict"], ela_r["verdict"], dft_r["verdict"]]
    if "FLAG" in verdicts:
        overall = "HIGH PROBABILITY OF MANIPULATION"
    elif "WARN" in verdicts:
        overall = "INDICATORS PRESENT — FURTHER INVESTIGATION RECOMMENDED"
    else:
        overall = "NO MANIPULATION INDICATORS DETECTED"

    lines.append(f"  OVERALL             : {overall}")
    lines.append("")

    # Stage 1
    lines.append(sep)
    lines.append("STAGE 1 — EXIF METADATA AUDIT")
    lines.append(thin)
    for field, val in exif_r.get("fields", {}).items():
        lines.append(f"  {field:<35}: {val}")
    lines.append("")
    if exif_r["flags"]:
        lines.append("  FLAGS:")
        for flag in exif_r["flags"]:
            lines.append(f"    [!] {flag}")
    else:
        lines.append("  No EXIF flags raised.")
    lines.append("")

    # Stage 2
    lines.append(sep)
    lines.append("STAGE 2 — ERROR LEVEL ANALYSIS (ELA)")
    lines.append(thin)
    lines.append(f"  Re-save quality : {ELA_QUALITY}")
    lines.append(f"  Mean ELA level  : {ela_r['mean']:.2f}")
    lines.append(f"  ELA std dev     : {ela_r['std']:.2f}")
    lines.append(f"  Peak ELA value  : {ela_r['peak']:.0f}")
    lines.append("")
    if ela_r["flags"]:
        lines.append("  FLAGS:")
        for flag in ela_r["flags"]:
            lines.append(f"    [!] {flag}")
    else:
        lines.append("  No ELA flags raised.")
    lines.append("")

    # Stage 3
    lines.append(sep)
    lines.append("STAGE 3 — DFT FREQUENCY ANALYSIS")
    lines.append(thin)
    lines.append(f"  Spectral median  : {dft_r['median_db']:.2f} dB")
    lines.append(f"  Spectral peak    : {dft_r['peak_db']:.2f} dB")
    lines.append(f"  Peak above median: {dft_r['peak_above']:.2f} dB  (threshold: {DFT_THRESHOLD} dB)")
    lines.append("")
    if dft_r["flags"]:
        lines.append("  FLAGS:")
        for flag in dft_r["flags"]:
            lines.append(f"    [!] {flag}")
    else:
        lines.append("  No DFT flags raised.")
    lines.append("")

    # Interpreter notes
    lines.append(sep)
    lines.append("INTERPRETATION NOTES")
    lines.append(thin)
    lines.append("  No single method is definitive. Convergent findings across")
    lines.append("  multiple stages constitute a forensically meaningful indicator.")
    lines.append("  This tool is designed for triage, not adjudication.")
    lines.append("  All findings should be documented and peer-reviewed before")
    lines.append("  use in legal, regulatory, or enforcement proceedings.")
    lines.append(sep)
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_console(exif_r, ela_r, dft_r, figure_path, report_path):
    sep  = "=" * 72
    thin = "-" * 72
    print()
    print(sep)
    print("  DEEPFAKE & IMAGE FORGERY DETECTION — RESULTS")
    print(sep)

    for stage, result in [
        ("EXIF Metadata Audit", exif_r),
        ("Error Level Analysis", ela_r),
        ("DFT Frequency Analysis", dft_r),
    ]:
        v   = result["verdict"]
        col = {"FLAG": "\033[91m", "WARN": "\033[93m", "PASS": "\033[92m"}.get(v, "")
        rst = "\033[0m"
        print(f"  {stage:<28}: {col}{v}{rst}")
        for flag in result.get("flags", []):
            print(f"    [!] {flag}")

    print(thin)
    verdicts = [exif_r["verdict"], ela_r["verdict"], dft_r["verdict"]]
    if "FLAG" in verdicts:
        print("  \033[91mOVERALL: HIGH PROBABILITY OF MANIPULATION\033[0m")
    elif "WARN" in verdicts:
        print("  \033[93mOVERALL: INDICATORS PRESENT — FURTHER INVESTIGATION RECOMMENDED\033[0m")
    else:
        print("  \033[92mOVERALL: NO MANIPULATION INDICATORS DETECTED\033[0m")
    print(thin)
    print(f"  Figure  saved: {figure_path}")
    print(f"  Report  saved: {report_path}")
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake & Image Forgery Detection — Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deepfake_detect.py --image Lab/suspect.jpg
  python deepfake_detect.py --image Lab/photo.jpeg
        """
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the JPEG image to analyse (e.g. Lab/suspect.jpg)"
    )
    args = parser.parse_args()

    path = args.image
    if not os.path.isfile(path):
        print(f"\n[ERROR] File not found: {path}")
        print("        Place your image in the Lab/ folder and try again.\n")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".jpg", ".jpeg"):
        print(f"\n[WARN] This tool is optimised for JPEG images. File is '{ext}'.")
        print("       ELA results may be unreliable for PNG/TIFF inputs.\n")

    base        = os.path.splitext(path)[0]
    figure_path = base + "_analysis.png"
    report_path = base + "_report.txt"

    print(f"\n[*] Analysing: {path}")

    print("[1/3] Running EXIF metadata audit...")
    exif_r = audit_exif(path)

    print("[2/3] Running Error Level Analysis...")
    ela_r = ela(path)

    print("[3/3] Running DFT frequency analysis...")
    dft_r = dft_analysis(path)

    print("[+]   Rendering forensic figure...")
    render_figure(path, exif_r, ela_r, dft_r, figure_path)

    print("[+]   Writing report...")
    write_report(path, exif_r, ela_r, dft_r, report_path)

    print_console(exif_r, ela_r, dft_r, figure_path, report_path)


if __name__ == "__main__":
    main()
