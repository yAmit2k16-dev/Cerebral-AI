import base64
import html
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image

# Optional PDF support — app still runs fully without reportlab.
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Optional OpenCV for faster colormap; pure-numpy fallback is always available.
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Cerebral AI | MRI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "best_vgg16_brain_tumor.keras"
TRAIN_DIR = BASE / "Training"

FALLBACK_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

if TRAIN_DIR.exists():
    CLASSES = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
else:
    CLASSES = FALLBACK_CLASSES

CLASS_INFO = {
    "glioma": "Tumours arising from glial cells, the supporting cells of the brain and spinal cord.",
    "meningioma": "Tumours growing from the meninges, the membranes that surround the brain.",
    "notumor": "No tumour pattern was recognised by the model in this scan.",
    "pituitary": "Growths in the pituitary gland at the base of the brain, which regulates hormones.",
}

# Richer educational narratives used only in the detailed assessment report.
CLASS_ASSESSMENT = {
    "glioma": {
        "overview": (
            "Gliomas are primary central-nervous-system tumours that originate from glial cells "
            "(astrocytes, oligodendrocytes or ependymal cells). They range from relatively slow-growing "
            "low-grade lesions to highly aggressive high-grade glioblastomas."
        ),
        "imaging_notes": (
            "On MRI, gliomas commonly appear as intra-axial masses with variable enhancement, "
            "surrounding oedema and possible mass effect. Exact characterisation requires multi-sequence "
            "review (T1, T2, FLAIR, contrast-enhanced T1 and often advanced sequences)."
        ),
        "next_steps": (
            "Clinical correlation with neurological examination, further multi-planar MRI sequences, "
            "and multidisciplinary review (neuroradiology / neurosurgery / oncology) is essential. "
            "Tissue diagnosis remains the gold standard when intervention is considered."
        ),
    },
    "meningioma": {
        "overview": (
            "Meningiomas arise from arachnoid cap cells of the meninges. Most are benign (WHO grade 1) "
            "and slow-growing; a minority are atypical or malignant. They are the most common primary "
            "intracranial tumours in adults."
        ),
        "imaging_notes": (
            "Typical MRI features include an extra-axial location, broad dural attachment (dural tail), "
            "homogeneous enhancement and possible hyperostosis of adjacent bone. Calcification may be present."
        ),
        "next_steps": (
            "Management depends on size, location, growth rate and symptoms. Observation with serial imaging "
            "is common for small asymptomatic lesions; larger or symptomatic tumours may require resection "
            "or radiosurgery after specialist review."
        ),
    },
    "notumor": {
        "overview": (
            "The model did not detect imaging patterns consistent with the trained tumour classes "
            "(glioma, meningioma or pituitary adenoma) on this scan."
        ),
        "imaging_notes": (
            "Absence of a model-detected tumour pattern does not exclude pathology. Technical factors "
            "(slice selection, contrast phase, motion artefact) or lesions outside the training distribution "
            "can produce a negative classification."
        ),
        "next_steps": (
            "Interpret the result only in the full clinical context. If symptoms or other imaging findings "
            "are present, formal radiologist review of the complete MRI study remains mandatory."
        ),
    },
    "pituitary": {
        "overview": (
            "Pituitary tumours (most often adenomas) arise within the sella turcica. They may be "
            "hormonally active or non-functioning and can cause mass effect on the optic chiasm or "
            "invade the cavernous sinuses."
        ),
        "imaging_notes": (
            "MRI of the sella with thin-slice, dynamic contrast-enhanced sequences is the modality of choice. "
            "Microadenomas (<10 mm) and macroadenomas have distinct imaging and clinical implications."
        ),
        "next_steps": (
            "Endocrine evaluation (pituitary hormone panel) and formal visual-field testing are typically "
            "indicated. Management may involve medical therapy, trans-sphenoidal surgery or observation, "
            "guided by a pituitary multidisciplinary team."
        ),
    },
}

# One colour per class, reused in the card, ring, donut and bars.
CLASS_COLORS = {
    "glioma": "#f59e4b",
    "meningioma": "#a78bfa",
    "notumor": "#43dbc5",
    "pituitary": "#60a5fa",
}
FALLBACK_PALETTE = ["#f59e4b", "#a78bfa", "#43dbc5", "#60a5fa", "#f472b6", "#facc15"]


def class_color(name: str) -> str:
    if name.lower() in CLASS_COLORS:
        return CLASS_COLORS[name.lower()]
    idx = CLASSES.index(name) if name in CLASSES else 0
    return FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def pretty(name: str) -> str:
    if name.lower() == "notumor":
        return "No Tumor"
    return name.replace("_", " ").title()


def block(s: str) -> str:
    """Flatten HTML so Markdown never treats indentation as a code block."""
    return "".join(line.strip() for line in s.splitlines() if line.strip())


def confidence_level(score: float) -> tuple[str, str]:
    if score >= 0.85:
        return "High", "#43dbc5"
    if score >= 0.60:
        return "Moderate", "#f5b942"
    return "Low", "#f87171"


def build_report_markdown(result: dict, file_name: str = "uploaded_scan") -> str:
    """Generate a detailed Markdown assessment report from model output."""
    name = result["name"]
    display = pretty(name)
    conf = float(result["score"])
    level, _ = confidence_level(conf)
    ranked = sorted(zip(CLASSES, result["scores"]), key=lambda p: p[1], reverse=True)
    assessment = CLASS_ASSESSMENT.get(
        name.lower(),
        {
            "overview": CLASS_INFO.get(name.lower(), "Highest-scoring category from the classifier."),
            "imaging_notes": "Refer to standard radiological criteria for this category.",
            "next_steps": "Correlate with clinical findings and obtain specialist review.",
        },
    )
    ts = result.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = result.get("seconds", 0.0)

    lines = [
        "# Cerebral AI — Detailed MRI Assessment Report",
        "",
        f"**Generated:** {ts}  ",
        f"**Source file:** `{file_name}`  ",
        f"**Model:** VGG16 fine-tuned classifier ({len(CLASSES)} classes)  ",
        f"**Inference time:** {elapsed:.3f} s",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"The model’s top prediction for this brain MRI is **{display}** "
        f"with a softmax confidence of **{conf * 100:.2f}%** ({level} confidence).",
        "",
        assessment["overview"],
        "",
        "## 2. Classification Result",
        "",
        f"| Field | Value |",
        f"| --- | --- |",
        f"| Predicted class | {display} |",
        f"| Softmax score | {conf * 100:.2f}% |",
        f"| Confidence level | {level} |",
        f"| Runner-up | {pretty(ranked[1][0]) if len(ranked) > 1 else '—'} "
        f"({ranked[1][1] * 100:.2f}%) |" if len(ranked) > 1 else "| Runner-up | — |",
        "",
        "## 3. Probability Distribution",
        "",
        "| Rank | Class | Probability |",
        "| ---: | --- | ---: |",
    ]
    for i, (cls, sc) in enumerate(ranked, 1):
        marker = " ← top" if i == 1 else ""
        lines.append(f"| {i} | {pretty(cls)} | {float(sc) * 100:.2f}%{marker} |")

    lines.extend(
        [
            "",
            "## 4. Imaging & Clinical Context (Educational)",
            "",
            "### Typical imaging features",
            "",
            assessment["imaging_notes"],
            "",
            "### Suggested next steps (educational only)",
            "",
            assessment["next_steps"],
            "",
            "## 5. Model & Method Notes",
            "",
            "- Architecture: VGG16 (ImageNet-pretrained backbone, fine-tuned for 4-class brain MRI).",
            "- Input size: 224 × 224 pixels, ImageNet-style preprocessing.",
            "- Output: Softmax probabilities over the training class set.",
            "- Scores are **not** calibrated clinical probabilities and do **not** constitute a diagnosis.",
            "",
            "## 6. Important Disclaimer",
            "",
            "This report is generated automatically by an educational demonstration system. "
            "It is **not** a medical device, **not** a substitute for a qualified radiologist’s report, "
            "and must **not** be used to guide diagnosis, treatment or any healthcare decision. "
            "Always consult a licensed medical professional for interpretation of MRI examinations.",
            "",
            "---",
            "",
            "*Cerebral AI · Educational project · TensorFlow / Keras · Streamlit*",
        ]
    )
    return "\n".join(lines)


def build_report_pdf(result: dict, file_name: str = "uploaded_scan") -> bytes | None:
    """Generate a professional PDF assessment report and return its bytes.
    Returns None if reportlab is not installed.
    """
    if not HAS_REPORTLAB:
        return None

    name = result["name"]
    display = pretty(name)
    conf = float(result["score"])
    level, _ = confidence_level(conf)
    ranked = sorted(zip(CLASSES, result["scores"]), key=lambda p: p[1], reverse=True)
    assessment = CLASS_ASSESSMENT.get(
        name.lower(),
        {
            "overview": CLASS_INFO.get(name.lower(), "Highest-scoring category from the classifier."),
            "imaging_notes": "Refer to standard radiological criteria for this category.",
            "next_steps": "Correlate with clinical findings and obtain specialist review.",
        },
    )
    ts = result.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = result.get("seconds", 0.0)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Cerebral AI Report — {display}",
        author="Cerebral AI",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#0c1b2e"),
            spaceAfter=4,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#4a6075"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#0c1b2e"),
            spaceBefore=14,
            spaceAfter=6,
            borderPadding=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1a2c40"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["Normal"],
            fontSize=8.5,
            textColor=colors.HexColor("#5a738c"),
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Disclaimer",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#6b3a00"),
            backColor=colors.HexColor("#fff8e6"),
            borderPadding=6,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FooterNote",
            parent=styles["Normal"],
            fontSize=7.5,
            textColor=colors.HexColor("#7a92a9"),
            alignment=TA_CENTER,
            spaceBefore=8,
        )
    )

    story = []
    story.append(Paragraph("CEREBRAL AI", styles["CoverTitle"]))
    story.append(Paragraph("Detailed Brain MRI Assessment Report", styles["CoverSub"]))
    story.append(
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#43dbc5"), spaceAfter=10)
    )

    story.append(Paragraph(f"<b>Generated:</b> {ts}", styles["Meta"]))
    story.append(Paragraph(f"<b>Source file:</b> {html.escape(file_name)}", styles["Meta"]))
    story.append(
        Paragraph(
            f"<b>Model:</b> VGG16 fine-tuned · {len(CLASSES)} classes · "
            f"Inference {elapsed:.3f} s",
            styles["Meta"],
        )
    )
    story.append(Spacer(1, 8))

    # Executive summary box
    summary_data = [
        [
            Paragraph("<b>Predicted class</b>", styles["BodyJust"]),
            Paragraph(f"<b>{display}</b>", styles["BodyJust"]),
        ],
        [
            Paragraph("<b>Softmax confidence</b>", styles["BodyJust"]),
            Paragraph(f"{conf * 100:.2f}%  ({level})", styles["BodyJust"]),
        ],
        [
            Paragraph("<b>Runner-up</b>", styles["BodyJust"]),
            Paragraph(
                f"{pretty(ranked[1][0])} ({ranked[1][1] * 100:.2f}%)"
                if len(ranked) > 1
                else "—",
                styles["BodyJust"],
            ),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[55 * mm, 110 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f7fb")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#2a4660")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d6e6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Executive Summary", styles["SectionHead"]))
    story.append(
        Paragraph(
            f"The model’s top prediction for this brain MRI is <b>{display}</b> "
            f"with a softmax confidence of <b>{conf * 100:.2f}%</b> ({level} confidence). "
            + assessment["overview"],
            styles["BodyJust"],
        )
    )

    story.append(Paragraph("2. Probability Distribution", styles["SectionHead"]))
    prob_header = [
        Paragraph("<b>Rank</b>", styles["BodyJust"]),
        Paragraph("<b>Class</b>", styles["BodyJust"]),
        Paragraph("<b>Probability</b>", styles["BodyJust"]),
    ]
    prob_rows = [prob_header]
    for i, (cls, sc) in enumerate(ranked, 1):
        mark = "  ← top" if i == 1 else ""
        prob_rows.append(
            [
                Paragraph(str(i), styles["BodyJust"]),
                Paragraph(pretty(cls) + mark, styles["BodyJust"]),
                Paragraph(f"{float(sc) * 100:.2f}%", styles["BodyJust"]),
            ]
        )
    prob_table = Table(prob_rows, colWidths=[20 * mm, 90 * mm, 40 * mm])
    prob_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0c1b2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e6f7f4")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#2a4660")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c5d6e6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(prob_table)

    story.append(Paragraph("3. Imaging & Clinical Context (Educational)", styles["SectionHead"]))
    story.append(Paragraph("<b>Typical imaging features</b>", styles["BodyJust"]))
    story.append(Paragraph(assessment["imaging_notes"], styles["BodyJust"]))
    story.append(Paragraph("<b>Suggested next steps (educational only)</b>", styles["BodyJust"]))
    story.append(Paragraph(assessment["next_steps"], styles["BodyJust"]))

    story.append(Paragraph("4. Model & Method Notes", styles["SectionHead"]))
    story.append(
        Paragraph(
            "• Architecture: VGG16 (ImageNet-pretrained backbone, fine-tuned for multi-class brain MRI).<br/>"
            "• Input: 224 × 224 RGB, ImageNet-style preprocessing.<br/>"
            "• Output: Softmax probabilities over the training class set.<br/>"
            "• Scores are <b>not</b> calibrated clinical probabilities and do <b>not</b> constitute a diagnosis.",
            styles["BodyJust"],
        )
    )

    story.append(
        Paragraph(
            "<b>IMPORTANT DISCLAIMER</b><br/><br/>"
            "This report is generated automatically by an educational demonstration system. "
            "It is <b>not</b> a medical device, <b>not</b> a substitute for a qualified radiologist’s report, "
            "and must <b>not</b> be used to guide diagnosis, treatment or any healthcare decision. "
            "Always consult a licensed medical professional for interpretation of MRI examinations.",
            styles["Disclaimer"],
        )
    )
    story.append(
        Paragraph(
            "Cerebral AI · Educational project · TensorFlow / Keras · Streamlit",
            styles["FooterNote"],
        )
    )

    doc.build(story)
    return buf.getvalue()


# =========================================================
# DESIGN SYSTEM
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

@property --angle { syntax:'<angle>'; initial-value:0deg; inherits:false; }

:root {
    --bg:#060d19;
    --border:#22364b;
    --border-soft:#1a2c40;
    --text:#edf4fc;
    --muted:#8fa6bd;
    --teal:#43dbc5;
    --teal-soft:rgba(67,219,197,.12);
    --amber:#f5b942;
    --glass:rgba(16,30,48,.62);
}

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
::selection { background:rgba(67,219,197,.35); color:#fff; }
::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-track { background:#08111f; }
::-webkit-scrollbar-thumb { background:#223a52; border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background:#2f5470; }

.stApp {
    background:
        radial-gradient(ellipse 55% 38% at 85% -8%, rgba(67,219,197,.13) 0%, transparent 70%),
        radial-gradient(ellipse 45% 40% at -5% 105%, rgba(96,130,255,.12) 0%, transparent 70%),
        radial-gradient(circle at 1px 1px, rgba(255,255,255,.045) 1px, transparent 0) 0 0 / 26px 26px,
        var(--bg);
    color:var(--text);
}

[data-testid="stHeader"] { background:rgba(6,13,25,.6); backdrop-filter:blur(10px); }
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0a1626,#070f1b);
    border-right:1px solid var(--border-soft);
}
[data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
.block-container { max-width:1400px; padding:1.5rem 2.4rem 2rem; }

h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.7px; color:var(--text) !important; }
p,label,.stMarkdown { color:var(--text); }
footer,#MainMenu { visibility:hidden; }
hr { border-color:var(--border-soft); margin:1.1rem 0; }

/* ---------- Sidebar ---------- */
.brand { display:flex; align-items:center; gap:12px; margin-bottom:18px; }
.brand-icon {
    width:48px; height:48px; display:flex; align-items:center; justify-content:center;
    background:linear-gradient(140deg,#1f6670,#153b51);
    border:1px solid #3a8d95; border-radius:15px; font-size:25px;
    box-shadow:0 8px 24px rgba(67,219,197,.2), inset 0 1px 0 rgba(255,255,255,.12);
}
.brand-name { color:#f3f8ff; font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; letter-spacing:1.4px; }
.brand-sub { color:var(--muted); font-size:10px; letter-spacing:1.9px; margin-top:2px; }
.pill-row { display:flex; gap:8px; flex-wrap:wrap; }
.pill {
    display:inline-flex; align-items:center; gap:8px;
    background:#0e2d2d; border:1px solid #236a5c; color:#72e8ce;
    padding:6px 12px; border-radius:20px; font-size:10px; font-weight:700; letter-spacing:.9px;
}
.pill.ghost { background:#0d1b2d; border-color:#27405a; color:#9db6cd; }
.pill .dot { width:7px; height:7px; border-radius:50%; background:var(--teal); box-shadow:0 0 8px var(--teal); animation:pulseDot 2s ease-in-out infinite; }

.side-head { color:#dce8f5; font-weight:700; font-size:11px; letter-spacing:1.5px; margin-bottom:10px; }
.side-copy { color:#89a1b9; font-size:12px; line-height:1.75; }
.spec-row { display:flex; justify-content:space-between; gap:10px; padding:9px 0; border-bottom:1px solid var(--border-soft); }
.spec-row span:first-child { color:#8fa7be; font-size:12px; }
.spec-row span:last-child { color:#e8f2fc; font-size:12px; font-weight:600; }
.class-item {
    display:flex; align-items:center; gap:11px; color:#c0d2e3; font-size:12.5px;
    padding:8px 10px; margin:3px 0; border-radius:10px; border:1px solid transparent;
    transition:background .2s ease, border-color .2s ease, transform .2s ease;
}
.class-item:hover { background:rgba(255,255,255,.04); border-color:var(--border); transform:translateX(3px); }
.class-item i { width:10px; height:10px; border-radius:4px; background:var(--c); box-shadow:0 0 10px var(--c); }

/* ---------- Hero ---------- */
.hero {
    background:linear-gradient(115deg,rgba(22,52,74,.92),rgba(11,26,44,.9));
    border:1px solid #2a4660; border-radius:24px; padding:32px 36px; margin:2px 0 18px;
    position:relative; overflow:hidden; box-shadow:0 24px 60px -24px rgba(0,0,0,.7);
    animation:panelReveal .6s ease-out both;
}
.hero:before {
    content:""; position:absolute; inset:0;
    background:radial-gradient(circle at 90% 10%, rgba(67,219,197,.18), transparent 45%);
    pointer-events:none;
}
.hero-art { position:absolute; right:-10px; top:-10px; width:360px; height:100%; opacity:.9; pointer-events:none; }
.hero-art .edge { stroke:rgba(67,219,197,.28); stroke-width:1; }
.hero-art .node { fill:#43dbc5; animation:nodePulse 3.2s ease-in-out infinite; }
.hero-art .ring { fill:none; stroke:rgba(67,219,197,.18); }
.hero-inner { position:relative; z-index:1; }
.eyebrow { color:var(--teal); font-size:11px; font-weight:700; letter-spacing:2.2px; }
.hero-title {
    font-family:'Space Grotesk',sans-serif; font-size:clamp(30px,3.4vw,46px);
    line-height:1.1; font-weight:700; margin:12px 0 12px;
    background:linear-gradient(90deg,#ffffff 20%,#9cf0e2 70%,#7fb6ff);
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero-copy { color:#a9bed2; font-size:14.5px; max-width:600px; line-height:1.7; }

/* ---------- Stepper ---------- */
.stepper { display:flex; gap:10px; flex-wrap:wrap; margin:0 0 24px; animation:panelReveal .6s .1s ease-out both; }
.st-chip {
    flex:1; min-width:170px; display:flex; align-items:center; gap:12px;
    background:var(--glass); border:1px solid var(--border-soft); border-radius:14px;
    padding:11px 14px; backdrop-filter:blur(8px); transition:all .3s ease;
}
.st-chip .num {
    width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:11px; font-weight:700; color:#6f88a0; background:#0d1b2d; border:1px solid #28405a; transition:all .3s ease;
}
.st-chip .lbl { font-size:12.5px; font-weight:600; color:#7f97ae; }
.st-chip .sub { font-size:10.5px; color:#5f7890; margin-top:1px; }
.st-chip.active { border-color:#2f6f78; background:rgba(19,54,63,.6); box-shadow:0 0 24px rgba(67,219,197,.12); }
.st-chip.active .num { color:#06211f; background:var(--teal); border-color:var(--teal); box-shadow:0 0 14px rgba(67,219,197,.6); }
.st-chip.active .lbl { color:#f0f8ff; }
.st-chip.done .num { color:var(--teal); background:var(--teal-soft); border-color:#2a6a68; }
.st-chip.done .lbl { color:#c4e9e3; }

/* ---------- Section heads ---------- */
.panel-top { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
.step {
    display:flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:10px;
    background:#13363f; border:1px solid #28606a; color:var(--teal); font-size:12px; font-weight:700;
}
.section-head { font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:600; color:#f0f6ff; }
.section-sub { color:var(--muted); font-size:11px; letter-spacing:1.2px; margin:2px 0 14px 44px; }

/* ---------- Glass panels ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(155deg,rgba(18,34,54,.78),rgba(10,21,36,.82));
    backdrop-filter:blur(10px);
    border:1px solid var(--border) !important; border-radius:20px !important;
    box-shadow:0 18px 44px -20px rgba(0,0,0,.75), inset 0 1px 0 rgba(255,255,255,.04);
    transition:border-color .3s ease, box-shadow .3s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:#2e4c68 !important; }

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] {
    background:#0a1627; border:1px dashed #36526b; border-radius:14px; padding:10px;
    transition:border-color .2s ease, background .2s ease, box-shadow .2s ease;
}
[data-testid="stFileUploader"]:hover { border-color:var(--teal); background:#0c1c30; box-shadow:0 0 22px rgba(67,219,197,.1); }
[data-testid="stFileUploader"] section { background:transparent; border:none; }
[data-testid="stFileUploader"] button {
    border-radius:9px; border:1px solid #2f6f78; background:rgba(67,219,197,.08); color:#bff6ec;
}
[data-testid="stFileUploader"] button:hover { border-color:var(--teal); color:#fff; }

.file-chip {
    display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
    background:#0a1627; border:1px solid var(--border-soft); border-radius:12px;
    padding:10px 14px; margin-top:12px; font-size:12px; color:#9db3c8;
}
.file-chip b { color:#eaf3fc; font-weight:600; overflow-wrap:anywhere; }

/* ---------- Scan viewer (HUD) ---------- */
.viewer {
    position:relative; overflow:hidden; border-radius:16px; background:#03080f;
    border:1px solid #2a4660; box-shadow:0 12px 34px -14px rgba(0,0,0,.8);
    margin-top:12px;
}
.viewer img { display:block; width:100%; max-height:440px; object-fit:contain; }
.viewer:before {
    content:""; position:absolute; inset:0; z-index:2; pointer-events:none;
    background:
        linear-gradient(rgba(67,219,197,.05) 1px, transparent 1px) 0 0 / 100% 5px,
        radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,.45) 100%);
}
.corner { position:absolute; width:22px; height:22px; border:2px solid var(--teal); z-index:3; opacity:.85; }
.corner.tl { top:10px; left:10px; border-right:none; border-bottom:none; border-top-left-radius:6px; }
.corner.tr { top:10px; right:10px; border-left:none; border-bottom:none; border-top-right-radius:6px; }
.corner.bl { bottom:10px; left:10px; border-right:none; border-top:none; border-bottom-left-radius:6px; }
.corner.br { bottom:10px; right:10px; border-left:none; border-top:none; border-bottom-right-radius:6px; }
.hud {
    position:absolute; z-index:3; font-size:10px; font-weight:700; letter-spacing:1.4px;
    color:rgba(67,219,197,.9); background:rgba(3,8,15,.6); padding:3px 8px; border-radius:6px;
}
.hud.l { left:42px; top:50%; transform:translateY(-50%); }
.hud.r { right:42px; top:50%; transform:translateY(-50%); }
.hud.b { left:50%; bottom:12px; transform:translateX(-50%); font-weight:500; color:#9ccfc8; }
.viewer.scanning { border-color:#2f8077; box-shadow:0 0 36px rgba(67,219,197,.25); }
.viewer.scanning img { opacity:.8; }
.scan-line {
    position:absolute; left:0; right:0; top:0; height:3px; z-index:4; background:var(--teal);
    box-shadow:0 0 20px 7px rgba(67,219,197,.55); animation:scanSweep 1.6s ease-in-out infinite;
}
.scan-tag {
    position:absolute; left:42px; top:14px; z-index:4; background:rgba(3,8,15,.85);
    border:1px solid #2f8077; color:var(--teal); font-size:10px; font-weight:700; letter-spacing:1.6px;
    padding:6px 11px; border-radius:20px; animation:blink 1.2s ease-in-out infinite;
}

/* ---------- Button ---------- */
.stButton > button {
    width:100%; min-height:52px; border-radius:14px; border:1px solid #5ce1cd;
    background:linear-gradient(100deg,#2fbfae,#55e3cb); color:#06211f; font-weight:700; letter-spacing:.4px;
    position:relative; overflow:hidden;
    transition:background .2s ease, box-shadow .2s ease, transform .2s ease;
}
.stButton > button:before {
    content:""; position:absolute; top:0; left:-120%; width:60%; height:100%;
    background:linear-gradient(100deg,transparent,rgba(255,255,255,.45),transparent);
    transform:skewX(-20deg); transition:left .6s ease;
}
.stButton > button:hover:before { left:140%; }
.stButton > button:hover {
    border-color:#a4fff0; color:#06211f; transform:translateY(-2px);
    background:linear-gradient(100deg,#45d6c4,#78f0dd); box-shadow:0 10px 30px rgba(67,219,197,.35);
}
.stButton > button:active { transform:translateY(0); }
.stButton > button:disabled { background:#122338; border-color:#25394e; color:#587088; box-shadow:none; transform:none; }
.stButton > button:disabled:before { display:none; }

/* ---------- Empty state ---------- */
.empty-state {
    background:#0a1627; border:1px dashed #33506a; border-radius:16px; text-align:center; padding:44px 18px;
}
.empty-icon {
    position:relative; width:64px; height:64px; margin:0 auto 16px; border-radius:20px;
    display:flex; align-items:center; justify-content:center;
    background:var(--teal-soft); border:1px solid #2a6a68; color:var(--teal); font-size:27px;
    animation:floaty 3.4s ease-in-out infinite;
}
.empty-icon:after {
    content:""; position:absolute; inset:-9px; border-radius:26px; border:1px solid rgba(67,219,197,.25);
    animation:ripple 3.4s ease-out infinite;
}
.empty-title { font-family:'Space Grotesk',sans-serif; font-weight:600; color:#edf5ff; font-size:17px; }
.empty-copy { color:#91a9bf; font-size:12.5px; line-height:1.8; margin-top:7px; }

/* ---------- Result card with animated border ---------- */
.result-card {
    display:flex; align-items:center; justify-content:space-between; gap:20px; position:relative;
    border:1.5px solid transparent; border-radius:20px; padding:24px 26px;
    background:
        radial-gradient(ellipse at 100% 0%, var(--tone-soft), transparent 65%) padding-box,
        linear-gradient(145deg,#0f2733,#0c1b2e) padding-box,
        conic-gradient(from var(--angle), rgba(255,255,255,.04) 0%, var(--tone) 18%, rgba(255,255,255,.04) 40%, rgba(255,255,255,.04) 100%) border-box;
    box-shadow:0 0 36px var(--tone-soft);
    animation:resultReveal .55s ease-out both, spin 5s linear infinite;
}
.result-main { flex:1; min-width:0; }
.result-label { display:inline-flex; align-items:center; gap:8px; color:var(--tone); font-size:10px; font-weight:700; letter-spacing:1.9px; }
.result-label:before { content:""; width:7px; height:7px; border-radius:50%; background:var(--tone); box-shadow:0 0 10px var(--tone); }
.result-name {
    color:#f5fbff; font-family:'Space Grotesk',sans-serif; font-size:clamp(26px,2.8vw,34px);
    font-weight:700; margin:8px 0; overflow-wrap:anywhere;
}
.result-note { color:#b2c8d3; font-size:12.5px; line-height:1.65; max-width:420px; }

.ring-wrap { position:relative; width:132px; height:132px; flex-shrink:0; }
.ring { width:100%; height:100%; }
.ring-bg { fill:none; stroke:rgba(255,255,255,.08); stroke-width:9; }
.ring-fg {
    fill:none; stroke:var(--tone); stroke-width:9; stroke-linecap:round;
    stroke-dasharray:326.73; stroke-dashoffset:var(--off);
    transform:rotate(-90deg); transform-origin:60px 60px; filter:drop-shadow(0 0 7px var(--tone));
    animation:ringFill 1.5s cubic-bezier(.2,.8,.2,1) both;
}
.ring-text { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.ring-num { font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; color:#f2f7ff; font-variant-numeric:tabular-nums; }
.ring-cap { color:#8fa7be; font-size:9px; letter-spacing:1.4px; margin-top:2px; }

/* ---------- Stat cards ---------- */
.stat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:14px 0 4px; }
.stat-card {
    background:rgba(11,24,40,.8); border:1px solid var(--border-soft); border-radius:14px; padding:14px 16px;
    animation:panelReveal .55s ease-out both; transition:transform .25s ease, border-color .25s ease;
}
.stat-card:hover { transform:translateY(-3px); border-color:#2f4d68; }
.stat-card:nth-child(2) { animation-delay:.08s; }
.stat-card:nth-child(3) { animation-delay:.16s; }
.stat-label { color:#8fa7be; text-transform:uppercase; letter-spacing:1.2px; font-size:9.5px; font-weight:700; }
.stat-value { font-family:'Space Grotesk',sans-serif; color:#f2f7ff; font-size:20px; font-weight:700; margin-top:6px; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
.stat-sub { color:#7f97ad; font-size:11px; margin-top:2px; }
.lvl-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; vertical-align:middle; }

.info-box {
    display:flex; gap:12px; align-items:flex-start; margin-top:14px;
    background:#17202c; border:1px solid var(--border-soft); border-left:3px solid var(--amber);
    border-radius:12px; padding:13px 15px; color:#b9c6d4; font-size:12.5px; line-height:1.65;
}
.info-box b { color:#fff3d6; }

/* ---------- Probabilities ---------- */
.prob-title { font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:600; color:#f0f6ff; margin-top:24px; }
.prob-sub { color:var(--muted); font-size:10.5px; letter-spacing:1.2px; margin:3px 0 12px; }
.prob-grid { display:grid; grid-template-columns:150px 1fr; gap:22px; align-items:center; }
.donut-wrap { position:relative; width:150px; height:150px; }
.donut { width:100%; height:100%; transform:rotate(-90deg); }
.donut circle.base { fill:none; stroke:#14263a; stroke-width:14; }
.donut circle.seg { fill:none; stroke-width:14; animation:segIn .9s ease-out both; }
.donut-center { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }
.donut-center b { font-family:'Space Grotesk',sans-serif; font-size:19px; color:#f2f7ff; }
.donut-center span { font-size:9px; letter-spacing:1.3px; color:#8fa7be; margin-top:2px; }

.prob-row { margin:13px 0; }
.prob-head { display:flex; justify-content:space-between; margin-bottom:7px; gap:8px; }
.prob-name { color:#c7d8e8; font-size:13px; font-weight:500; display:flex; align-items:center; gap:9px; }
.prob-name i { width:9px; height:9px; border-radius:3px; background:var(--c); box-shadow:0 0 8px var(--c); }
.prob-pct { color:#8fa7be; font-size:12px; font-variant-numeric:tabular-nums; }
.prob-row.top .prob-name { color:#fff; font-weight:700; }
.prob-row.top .prob-pct { color:var(--c); font-weight:700; }
.prob-track { height:9px; background:#16283c; border-radius:20px; overflow:hidden; }
.prob-fill {
    height:100%; width:var(--w); border-radius:20px;
    background:linear-gradient(90deg, rgba(255,255,255,.0), rgba(255,255,255,.0)), var(--c);
    opacity:.55; animation:barGrow 1s cubic-bezier(.2,.8,.2,1) both; animation-delay:var(--d);
}
.prob-row.top .prob-fill { opacity:1; box-shadow:0 0 14px var(--c); }
.fine-print { color:#7a92a9; font-size:11px; line-height:1.7; margin-top:16px; }

/* ---------- Footer ---------- */
.footer {
    text-align:center; color:#7f97ae; font-size:11px; line-height:1.9; padding:16px 0 6px;
}
.footer b { color:#b9cadb; letter-spacing:1.4px; }
.footer .tags { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-bottom:10px; }
.footer .tag { border:1px solid var(--border); background:rgba(255,255,255,.02); padding:4px 11px; border-radius:20px; font-size:10px; letter-spacing:.9px; color:#9db3c8; }

/* ---------- Animations ---------- */
@keyframes spin { to { --angle:360deg; } }
@keyframes scanSweep { 0%{top:0%;opacity:0;} 10%{opacity:1;} 90%{opacity:1;} 100%{top:100%;opacity:0;} }
@keyframes panelReveal { from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:translateY(0);} }
@keyframes resultReveal { from{opacity:0;transform:translateY(14px) scale(.985);} to{opacity:1;transform:translateY(0) scale(1);} }
@keyframes ringFill { from{stroke-dashoffset:326.73;} }
@keyframes barGrow { from{width:0;} }
@keyframes segIn { from{opacity:0;stroke-width:4;} }
@keyframes pulseDot { 0%,100%{opacity:.5;transform:scale(.85);} 50%{opacity:1;transform:scale(1.15);} }
@keyframes nodePulse { 0%,100%{opacity:.35;} 50%{opacity:1;} }
@keyframes floaty { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-5px);} }
@keyframes ripple { 0%{transform:scale(.85);opacity:.8;} 100%{transform:scale(1.35);opacity:0;} }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.45;} }

.report-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }

@media (max-width:900px) {
    .block-container { padding:1rem 1rem 2rem; }
    .hero { padding:24px 20px; }
    .hero-art { display:none; }
    .result-card { flex-direction:column-reverse; align-items:flex-start; }
    .stat-grid { grid-template-columns:1fr; }
    .prob-grid { grid-template-columns:1fr; justify-items:center; }
    .prob-grid > div:last-child { width:100%; }
    .report-grid { grid-template-columns:1fr; }
}
@media (prefers-reduced-motion: reduce) {
    *, *:before, *:after { animation:none !important; transition:none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        block(
            """
        <div class="brand">
          <div class="brand-icon">🧠</div>
          <div>
            <div class="brand-name">CEREBRAL</div>
            <div class="brand-sub">AI INTELLIGENCE</div>
          </div>
        </div>
        <div class="pill-row">
          <div class="pill"><span class="dot"></span>LOCAL MODEL</div>
          <div class="pill ghost">v2.0</div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        block(
            """
        <div class="side-head">PROJECT OVERVIEW</div>
        <div class="side-copy">A deep-learning demonstration for classifying
        brain MRI images using a fine-tuned VGG16 convolutional neural network.</div>
        """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    specs = [
        ("Architecture", "VGG16"),
        ("Input resolution", "224 × 224"),
        ("Classification", f"{len(CLASSES)} classes"),
        ("Framework", "TensorFlow / Keras"),
    ]
    rows = "".join(
        f'<div class="spec-row"><span>{html.escape(k)}</span><span>{html.escape(v)}</span></div>'
        for k, v in specs
    )
    st.markdown(f'<div class="side-head">MODEL SPECIFICATIONS</div>{rows}', unsafe_allow_html=True)

    st.markdown("---")
    items = "".join(
        f'<div class="class-item" style="--c:{class_color(n)}"><i></i>{html.escape(pretty(n))}</div>'
        for n in CLASSES
    )
    st.markdown(f'<div class="side-head">OUTPUT CATEGORIES</div>{items}', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        block(
            """
        <div class="side-copy">ACADEMIC PROJECT<br>AI / MACHINE LEARNING<br><br>
        For educational demonstration only.</div>
        """
        ),
        unsafe_allow_html=True,
    )

# =========================================================
# HERO
# =========================================================
HERO_ART = """
<svg class="hero-art" viewBox="0 0 360 220" preserveAspectRatio="xMidYMid slice">
  <circle class="ring" cx="250" cy="100" r="60"/><circle class="ring" cx="250" cy="100" r="95"/>
  <g class="edge">
    <line x1="170" y1="60" x2="240" y2="40"/><line x1="240" y1="40" x2="310" y2="80"/>
    <line x1="170" y1="60" x2="215" y2="115"/><line x1="215" y1="115" x2="310" y2="80"/>
    <line x1="215" y1="115" x2="280" y2="160"/><line x1="310" y1="80" x2="330" y2="150"/>
    <line x1="280" y1="160" x2="330" y2="150"/><line x1="240" y1="40" x2="215" y2="115"/>
    <line x1="150" y1="140" x2="215" y2="115"/><line x1="150" y1="140" x2="205" y2="190"/>
    <line x1="205" y1="190" x2="280" y2="160"/>
  </g>
  <g>
    <circle class="node" cx="170" cy="60" r="4"/><circle class="node" cx="240" cy="40" r="5" style="animation-delay:.4s"/>
    <circle class="node" cx="310" cy="80" r="4" style="animation-delay:.8s"/><circle class="node" cx="215" cy="115" r="6" style="animation-delay:1.2s"/>
    <circle class="node" cx="280" cy="160" r="4" style="animation-delay:1.6s"/><circle class="node" cx="330" cy="150" r="3" style="animation-delay:2s"/>
    <circle class="node" cx="150" cy="140" r="3" style="animation-delay:2.4s"/><circle class="node" cx="205" cy="190" r="4" style="animation-delay:2.8s"/>
  </g>
</svg>
"""

st.markdown(
    block(
        f"""
<div class="hero">
  {HERO_ART}
  <div class="hero-inner">
    <div class="eyebrow">ARTIFICIAL INTELLIGENCE &nbsp;/&nbsp; MEDICAL IMAGING</div>
    <div class="hero-title">Brain MRI Analysis</div>
    <div class="hero-copy">An intelligent image-classification workspace. Upload an MRI scan
    to view your trained model's predicted class and probability scores.</div>
  </div>
</div>
"""
    ),
    unsafe_allow_html=True,
)

stepper_slot = st.empty()


def render_stepper(has_upload: bool, has_result: bool):
    states = ["", "", ""]
    done = [has_upload, has_result, has_result]
    first_pending = next((i for i, d in enumerate(done) if not d), None)
    for i in range(3):
        if done[i]:
            states[i] = "done"
        elif i == first_pending:
            states[i] = "active"
    steps = [
        ("Upload scan", "JPG or PNG image"),
        ("Run analysis", "VGG16 inference"),
        ("Review results", "Scores and charts"),
    ]
    chips = "".join(
        f'<div class="st-chip {states[i]}"><div class="num">{"✓" if states[i] == "done" else i + 1}</div>'
        f'<div><div class="lbl">{t}</div><div class="sub">{s}</div></div></div>'
        for i, (t, s) in enumerate(steps)
    )
    stepper_slot.markdown(f'<div class="stepper">{chips}</div>', unsafe_allow_html=True)


def scroll_to_top():
    """Scrolls the Streamlit page back to the top (runs once per call)."""
    components.html(
        f"""
        <script>
        // run id: {time.time()}
        const go = (behavior) => {{
            const d = window.parent.document;
            const targets = [
                d.querySelector('[data-testid="stMain"]'),
                d.querySelector('section.main'),
                d.querySelector('[data-testid="stAppViewContainer"]'),
                d.scrollingElement, d.documentElement, d.body
            ];
            targets.forEach(el => {{
                if (el && el.scrollTop > 0) {{
                    try {{ el.scrollTo({{top: 0, behavior}}); }} catch (e) {{ el.scrollTop = 0; }}
                }}
            }});
        }};
        go('smooth');
        setTimeout(() => go('auto'), 350);
        setTimeout(() => go('auto'), 800);
        </script>
        """,
        height=0,
    )


# =========================================================
# GRAD-CAM (explainability)
# =========================================================
def _layer_output_shape(layer):
    """Best-effort 4D output shape for a layer (handles nested quirks)."""
    try:
        shape = layer.output.shape
    except Exception:
        try:
            shape = layer.output_shape
        except Exception:
            return None
    if isinstance(shape, list):
        shape = shape[0]
    if shape is None:
        return None
    # Convert TensorShape → tuple
    try:
        shape = tuple(shape.as_list()) if hasattr(shape, "as_list") else tuple(shape)
    except Exception:
        return None
    return shape


def find_last_conv_layer(model):
    """
    Return (layer_name, layer_object) for the last spatial conv feature map.
    Searches top-level layers and one level of nested models (e.g. embedded VGG16).
    """
    candidates = []  # (depth_order, name, layer)

    def consider(layer, prefix=""):
        name = f"{prefix}{layer.name}" if prefix else layer.name
        shape = _layer_output_shape(layer)
        if shape is not None and len(shape) == 4:
            # Prefer layers that still have spatial size > 1
            spatial = shape[1] * shape[2] if shape[1] and shape[2] else 0
            candidates.append((spatial, name, layer))

    for layer in model.layers:
        consider(layer)
        # Nested model / backbone (common in transfer learning)
        if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
            try:
                for sub in layer.layers:
                    consider(sub, prefix=f"{layer.name}/")
            except Exception:
                pass

    if not candidates:
        # Hard-coded fallbacks for classic VGG16 fine-tunes
        for candidate in (
            "block5_conv3",
            "vgg16/block5_conv3",
            "conv2d_13",
            "conv5_block3_3_conv",
        ):
            try:
                if "/" in candidate:
                    parent, child = candidate.split("/", 1)
                    layer = model.get_layer(parent).get_layer(child)
                else:
                    layer = model.get_layer(candidate)
                return candidate, layer
            except Exception:
                continue
        raise ValueError(
            "No convolutional layer found for Grad-CAM. "
            f"Top-level layers: {[l.name for l in model.layers[:12]]}"
        )

    # Prefer the deepest / smallest spatial map among conv layers (typical last conv)
    candidates.sort(key=lambda x: x[0])  # smaller spatial size → later in VGG
    # Among smallest spatial, take the last encountered with that size
    min_spatial = candidates[0][0]
    best = [c for c in candidates if c[0] == min_spatial][-1]
    return best[1], best[2]


def _build_gradcam_model(model, target_conv_name: str = "block5_conv3"):
    """
    Rebuild a connected graph from the outer input → (target conv, predictions).

    Handles the common transfer-learning layout:
      input → sequential(augmentation) → vgg16(backbone) → head
    where a simple Model(inputs, [nested.output, model.output]) fails with
    "Output is not connected to inputs".
    """
    if hasattr(model, "input") and model.input is not None:
        inputs = model.input
    else:
        inputs = model.inputs[0] if isinstance(model.inputs, (list, tuple)) else model.inputs

    x = inputs
    conv_tensor = None
    used_name = target_conv_name

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        # Nested backbone (e.g. "vgg16") — walk internal layers to tap block5_conv3
        is_nested = isinstance(layer, tf.keras.Model) or (
            hasattr(layer, "layers")
            and type(layer).__name__ in ("Functional", "Sequential", "Model")
        )
        has_target = is_nested and any(
            getattr(s, "name", "") == target_conv_name
            for s in getattr(layer, "layers", [])
        )
        known_backbone = is_nested and layer.name.lower() in {
            "vgg16", "vgg19", "resnet50", "mobilenet", "efficientnetb0",
        }
        if has_target or known_backbone:
            y = x
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.InputLayer):
                    continue
                y = sub(y)
                if sub.name == target_conv_name:
                    conv_tensor = y
                    used_name = f"{layer.name}/{sub.name}"
            x = y
        else:
            x = layer(x)

    if conv_tensor is None:
        raise ValueError(
            f"Could not connect conv layer '{target_conv_name}' in the model graph. "
            "Expected a nested VGG16 with block5_conv3."
        )

    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_tensor, x], name="gradcam_model")
    return grad_model, used_name


def compute_gradcam(
    model,
    preprocessed_array: np.ndarray,
    class_index: int,
    layer_name: str | None = None,
):
    """
    Compute a Grad-CAM heatmap for the given class index.
    Returns (heatmap ndarray in [0,1], layer_name_used).
    """
    target = "block5_conv3"
    if layer_name:
        target = layer_name.split("/")[-1]

    grad_model, used_name = _build_gradcam_model(model, target_conv_name=target)

    img = tf.convert_to_tensor(preprocessed_array)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img, training=False)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Gradient is None — target layer may be disconnected from the loss. "
            f"Layer tried: {used_name}"
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)
    return heatmap.numpy(), used_name


def _numpy_jet_colormap(heatmap_uint8: np.ndarray) -> np.ndarray:
    """Simple jet-like colormap (H, W) uint8 → (H, W, 3) uint8 RGB without OpenCV."""
    x = heatmap_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    # Jet is darker at low values
    r = np.where(x < 0.125, 0.5 * x / 0.125 * r, r)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_gradcam(
    original_rgb: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Resize heatmap to the original image size, colourise it, and blend.
    Returns an RGB PIL Image.
    """
    heat = np.clip(heatmap, 0.0, 1.0)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
        original_rgb.size, resample=Image.BICUBIC
    )
    heat_np = np.asarray(heat_img)

    if HAS_CV2:
        colored = cv2.applyColorMap(heat_np, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        colored = _numpy_jet_colormap(heat_np)

    base = np.asarray(original_rgb).astype(np.float32)
    overlay = colored.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * overlay
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def heatmap_to_image(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Colourised heatmap only (no original image), resized to size."""
    heat = np.clip(heatmap, 0.0, 1.0)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
        size, resample=Image.BICUBIC
    )
    heat_np = np.asarray(heat_img)
    if HAS_CV2:
        colored = cv2.applyColorMap(heat_np, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        colored = _numpy_jet_colormap(heat_np)
    return Image.fromarray(colored)


# =========================================================
# MODEL
# =========================================================
@st.cache_resource(show_spinner="Loading trained model…")
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None
if "result" not in st.session_state:
    st.session_state.result = None


def image_to_b64(img: Image.Image, max_side: int = 1000) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def viewer_html(b64: str, size_label: str, scanning: bool = False) -> str:
    extra = (
        '<div class="scan-line"></div><div class="scan-tag">● ANALYZING SCAN</div>'
        if scanning
        else ""
    )
    return block(
        f"""
    <div class="viewer {'scanning' if scanning else ''}">
      <img src="data:image/jpeg;base64,{b64}">
      <span class="corner tl"></span><span class="corner tr"></span>
      <span class="corner bl"></span><span class="corner br"></span>
      <span class="hud l">L</span><span class="hud r">R</span>
      <span class="hud b">{html.escape(size_label)}</span>
      {extra}
    </div>
    """
    )


# =========================================================
# MAIN TWO-COLUMN WORKSPACE
# =========================================================
upload_col, result_col = st.columns([0.94, 1.06], gap="large")

with upload_col:
    st.markdown(
        block(
            """
        <div class="panel-top"><div class="step">01</div><div class="section-head">MRI Scan</div></div>
        <div class="section-sub">UPLOAD &nbsp;/&nbsp; IMAGE PREVIEW</div>
        """
        ),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploaded = st.file_uploader(
            "Select a brain MRI image",
            type=["jpg", "jpeg", "png"],
            help="Choose a JPG or PNG image from your Testing dataset.",
        )

        preview_slot = st.empty()
        info_slot = st.empty()
        preview = None
        preview_b64 = None
        size_label = ""

        if uploaded is not None:
            image_id = (uploaded.name, uploaded.size)
            if image_id != st.session_state.last_upload_id:
                st.session_state.last_upload_id = image_id
                st.session_state.result = None

            try:
                preview = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            except Exception:
                st.error("This image could not be opened. Please try another.")
                st.stop()

            preview_b64 = image_to_b64(preview)
            size_label = f"{preview.width} × {preview.height} px"
            preview_slot.markdown(viewer_html(preview_b64, size_label), unsafe_allow_html=True)
            info_slot.markdown(
                block(
                    f"""
                <div class="file-chip">
                  <span>FILE &nbsp;<b>{html.escape(uploaded.name)}</b></span>
                  <span>RESOLUTION &nbsp;<b>{preview.width} × {preview.height}</b></span>
                </div>
                """
                ),
                unsafe_allow_html=True,
            )
        else:
            preview_slot.markdown(
                block(
                    """
                <div class="empty-state">
                  <div class="empty-icon">⌁</div>
                  <div class="empty-title">Your scan goes here</div>
                  <div class="empty-copy">Choose an MRI image using the uploader above.<br>
                  Supported formats: JPG, JPEG and PNG.</div>
                </div>
                """
                ),
                unsafe_allow_html=True,
            )

    run_analysis = st.button("✦   Analyze MRI Scan", type="primary", disabled=uploaded is None)

with result_col:
    st.markdown(
        block(
            """
        <div class="panel-top"><div class="step">02</div><div class="section-head">Analysis Results</div></div>
        <div class="section-sub">CLASSIFICATION &nbsp;/&nbsp; MODEL OUTPUT</div>
        """
        ),
        unsafe_allow_html=True,
    )

    if run_analysis and uploaded is not None and preview is not None:
        try:
            preview_slot.markdown(
                viewer_html(preview_b64, size_label, scanning=True), unsafe_allow_html=True
            )

            started = time.perf_counter()
            image = preview.resize((224, 224))
            array = np.asarray(image, dtype=np.float32)
            array = np.expand_dims(array, axis=0)
            array = tf.keras.applications.vgg16.preprocess_input(array)
            scores = model.predict(array, verbose=0)[0]
            elapsed = time.perf_counter() - started

            if len(scores) != len(CLASSES):
                preview_slot.markdown(viewer_html(preview_b64, size_label), unsafe_allow_html=True)
                st.error(
                    "Model output count does not match the Training "
                    "class folders. Please check the class order."
                )
                st.stop()

            idx = int(np.argmax(scores))

            # Grad-CAM explainability map for the predicted class
            gradcam_overlay_b64 = None
            gradcam_heat_b64 = None
            gradcam_layer = None
            gradcam_error = None
            try:
                heatmap, layer_name = compute_gradcam(model, array, idx)
                overlay_img = overlay_gradcam(preview, heatmap, alpha=0.48)
                heat_only = heatmap_to_image(heatmap, preview.size)
                gradcam_overlay_b64 = image_to_b64(overlay_img)
                gradcam_heat_b64 = image_to_b64(heat_only)
                gradcam_layer = layer_name
            except Exception as gc_err:
                # Explainability is best-effort; classification still succeeds
                gradcam_error = f"{type(gc_err).__name__}: {gc_err}"

            time.sleep(max(0.0, 1.3 - elapsed))
            preview_slot.markdown(viewer_html(preview_b64, size_label), unsafe_allow_html=True)

            st.session_state.result = {
                "name": CLASSES[idx],
                "score": float(scores[idx]),
                "scores": scores.tolist(),
                "seconds": elapsed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": uploaded.name if uploaded is not None else "uploaded_scan",
                "image_size": f"{preview.width} × {preview.height}",
                "gradcam_overlay_b64": gradcam_overlay_b64,
                "gradcam_heat_b64": gradcam_heat_b64,
                "gradcam_layer": gradcam_layer,
                "gradcam_error": gradcam_error,
            }
            st.toast("Analysis complete", icon="✅")
            st.session_state.scroll_top = True
        except Exception as e:
            if preview_b64:
                preview_slot.markdown(viewer_html(preview_b64, size_label), unsafe_allow_html=True)
            st.error(f"Analysis failed: {e}")

    result = st.session_state.result

    if result is None:
        with st.container(border=True):
            st.markdown(
                block(
                    f"""
                <div class="empty-state">
                  <div class="empty-icon">⌗</div>
                  <div class="empty-title">Awaiting MRI scan</div>
                  <div class="empty-copy">Your results will appear here after analysis.<br>
                  The model will compare the image across {len(CLASSES)} classes.</div>
                </div>
                """
                ),
                unsafe_allow_html=True,
            )
    else:
        name = result["name"]
        display_name = pretty(name)
        confidence = result["score"]
        tone = class_color(name)

        ranked = sorted(zip(CLASSES, result["scores"]), key=lambda p: p[1], reverse=True)
        runner_name, runner_score = ranked[1] if len(ranked) > 1 else ("-", 0.0)

        offset = 326.73 * (1 - max(0.0, min(1.0, confidence)))
        description = CLASS_INFO.get(
            name.lower(), "Highest-scoring category produced by the trained classifier."
        )

        if confidence >= 0.85:
            level, lvl_color = "High", "#43dbc5"
        elif confidence >= 0.60:
            level, lvl_color = "Moderate", "#f5b942"
        else:
            level, lvl_color = "Low", "#f87171"

        st.markdown(
            block(
                f"""
            <div class="result-card" style="--tone:{tone};--tone-soft:{rgba(tone, .14)};--off:{offset:.2f}">
              <div class="result-main">
                <div class="result-label">MODEL PREDICTION</div>
                <div class="result-name">{html.escape(display_name)}</div>
                <div class="result-note">{html.escape(description)}</div>
              </div>
              <div class="ring-wrap">
                <svg class="ring" viewBox="0 0 120 120">
                  <circle class="ring-bg" cx="60" cy="60" r="52"/>
                  <circle class="ring-fg" cx="60" cy="60" r="52"/>
                </svg>
                <div class="ring-text">
                  <div class="ring-num">{confidence * 100:.1f}%</div>
                  <div class="ring-cap">TOP SCORE</div>
                </div>
              </div>
            </div>
            <div class="stat-grid">
              <div class="stat-card">
                <div class="stat-label">Score level</div>
                <div class="stat-value"><span class="lvl-dot" style="background:{lvl_color};box-shadow:0 0 10px {lvl_color}"></span>{level}</div>
                <div class="stat-sub">Top-class softmax</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Runner-up</div>
                <div class="stat-value">{html.escape(pretty(runner_name))}</div>
                <div class="stat-sub">{runner_score * 100:.2f}%</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Inference time</div>
                <div class="stat-value">{result.get('seconds', 0):.2f}s</div>
                <div class="stat-sub">{len(CLASSES)} classes evaluated</div>
              </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

        if level == "Low":
            st.markdown(
                block(
                    """
                <div class="info-box"><div>⚠</div>
                <div><b>Low model score.</b> The classifier is not confident about this
                image. Treat the result with extra caution.</div></div>
                """
                ),
                unsafe_allow_html=True,
            )

        # Donut chart segments
        C = 251.33  # 2 * pi * 40
        cum = 0.0
        segs = ""
        for i, (cls, sc) in enumerate(ranked):
            length = max(0.0, min(1.0, float(sc))) * C
            if length < 0.5:
                continue
            gap = 2.0 if length > 6 else 0.0
            segs += (
                f'<circle class="seg" cx="50" cy="50" r="40" stroke="{class_color(cls)}" '
                f'stroke-dasharray="{max(length - gap, 0.1):.2f} {C - max(length - gap, 0.1):.2f}" '
                f'stroke-dashoffset="{-cum:.2f}" style="animation-delay:{0.12 * i:.2f}s"/>'
            )
            cum += length

        donut = (
            f'<div class="donut-wrap"><svg class="donut" viewBox="0 0 100 100">'
            f'<circle class="base" cx="50" cy="50" r="40"/>{segs}</svg>'
            f'<div class="donut-center"><b>{confidence * 100:.0f}%</b>'
            f"<span>{html.escape(display_name.upper())}</span></div></div>"
        )

        bars = ""
        for i, (cls, sc) in enumerate(ranked):
            pct = max(0.0, min(1.0, float(sc)))
            row_cls = "prob-row top" if i == 0 else "prob-row"
            bars += (
                f'<div class="{row_cls}" style="--c:{class_color(cls)}">'
                f'<div class="prob-head"><span class="prob-name"><i></i>{html.escape(pretty(cls))}</span>'
                f'<span class="prob-pct">{pct * 100:.2f}%</span></div>'
                f'<div class="prob-track"><div class="prob-fill" '
                f'style="--w:{pct * 100:.2f}%;--d:{0.12 * i:.2f}s"></div></div></div>'
            )

        st.markdown(
            f'<div class="prob-title">Class probabilities</div>'
            f'<div class="prob-sub">SOFTMAX OUTPUTS · ALL CATEGORIES</div>'
            f'<div class="prob-grid">{donut}<div>{bars}</div></div>'
            f'<div class="fine-print">Model scores are not calibrated medical '
            f"probabilities and do not establish a diagnosis.</div>",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # GRAD-CAM EXPLAINABILITY
        # -------------------------------------------------
        gc_overlay = result.get("gradcam_overlay_b64")
        gc_heat = result.get("gradcam_heat_b64")
        gc_layer = result.get("gradcam_layer") or "last conv layer"

        st.markdown(
            block(
                f"""
            <div class="prob-title" style="margin-top:28px">Explainability · Grad-CAM</div>
            <div class="prob-sub">WHERE THE MODEL FOCUSED · LAYER {html.escape(str(gc_layer).upper())}</div>
            """
            ),
            unsafe_allow_html=True,
        )

        if gc_overlay:
            view_mode = st.radio(
                "Grad-CAM view",
                options=["Overlay", "Heatmap only", "Original"],
                index=0,
                horizontal=True,
                key="gradcam_view_mode",
                label_visibility="collapsed",
            )
            if view_mode == "Overlay":
                show_b64 = gc_overlay
                caption = "Grad-CAM overlay — warmer colours indicate regions that most influenced the prediction."
            elif view_mode == "Heatmap only":
                show_b64 = gc_heat or gc_overlay
                caption = "Raw Grad-CAM heatmap (jet colormap)."
            else:
                show_b64 = preview_b64 if preview_b64 else gc_overlay
                caption = "Original MRI scan."

            st.markdown(
                viewer_html(show_b64, size_label or result.get("image_size", "")),
                unsafe_allow_html=True,
            )
            st.caption(caption)
            st.markdown(
                block(
                    """
                <div class="fine-print" style="margin-top:8px">
                Grad-CAM highlights spatial regions the network used for its decision.
                It is an explanatory tool, not a lesion segmentation or clinical localisation map.
                </div>
                """
                ),
                unsafe_allow_html=True,
            )
        else:
            err_text = result.get("gradcam_error") or "Unknown error during Grad-CAM."
            st.markdown(
                block(
                    f"""
                <div class="info-box"><div>ℹ</div>
                <div><b>Grad-CAM unavailable for this run.</b>
                Classification results above are still valid.<br><br>
                <span style="color:#9db3c8;font-size:11px">{html.escape(str(err_text))}</span>
                </div></div>
                """
                ),
                unsafe_allow_html=True,
            )
            with st.expander("Model layer names (debug)"):
                try:
                    lines = []
                    for i, layer in enumerate(model.layers):
                        shape = _layer_output_shape(layer)
                        lines.append(f"{i:3d}  {layer.name:40s}  {shape}")
                        if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
                            try:
                                for j, sub in enumerate(layer.layers):
                                    sshape = _layer_output_shape(sub)
                                    lines.append(f"    .{j:02d}  {sub.name:36s}  {sshape}")
                            except Exception:
                                pass
                    st.code("\n".join(lines) if lines else "No layers found.")
                except Exception as e:
                    st.code(f"Could not list layers: {e}")

        # -------------------------------------------------
        # DETAILED ASSESSMENT REPORT
        # -------------------------------------------------
        assessment = CLASS_ASSESSMENT.get(
            name.lower(),
            {
                "overview": description,
                "imaging_notes": "Refer to standard radiological criteria for this category.",
                "next_steps": "Correlate with clinical findings and obtain specialist review.",
            },
        )
        file_name = result.get("file_name", "uploaded_scan")
        ts = result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        st.markdown(
            block(
                f"""
            <div class="prob-title" style="margin-top:28px">Detailed Assessment Report</div>
            <div class="prob-sub">EDUCATIONAL SUMMARY · DOWNLOADABLE PDF / MARKDOWN</div>
            """
            ),
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                block(
                    f"""
                <div style="padding:4px 2px 8px">
                  <div style="display:flex;flex-wrap:wrap;gap:10px 18px;margin-bottom:14px;font-size:12px;color:#9db3c8">
                    <span>FILE <b style="color:#eaf3fc">{html.escape(file_name)}</b></span>
                    <span>GENERATED <b style="color:#eaf3fc">{html.escape(ts)}</b></span>
                    <span>CONFIDENCE <b style="color:#eaf3fc">{confidence * 100:.1f}% · {level}</b></span>
                  </div>
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:600;color:#f0f6ff;margin-bottom:8px">
                    {html.escape(display_name)}
                  </div>
                  <div style="color:#b2c8d3;font-size:13px;line-height:1.7;margin-bottom:14px">
                    {html.escape(assessment["overview"])}
                  </div>
                  <div class="report-grid">
                    <div style="background:rgba(11,24,40,.8);border:1px solid var(--border-soft);border-radius:12px;padding:12px 14px">
                      <div style="color:#8fa7be;font-size:10px;font-weight:700;letter-spacing:1.1px;margin-bottom:6px">IMAGING NOTES</div>
                      <div style="color:#c7d8e8;font-size:12.5px;line-height:1.65">{html.escape(assessment["imaging_notes"])}</div>
                    </div>
                    <div style="background:rgba(11,24,40,.8);border:1px solid var(--border-soft);border-radius:12px;padding:12px 14px">
                      <div style="color:#8fa7be;font-size:10px;font-weight:700;letter-spacing:1.1px;margin-bottom:6px">SUGGESTED NEXT STEPS</div>
                      <div style="color:#c7d8e8;font-size:12.5px;line-height:1.65">{html.escape(assessment["next_steps"])}</div>
                    </div>
                  </div>
                </div>
                """
                ),
                unsafe_allow_html=True,
            )

            md_report = build_report_markdown(result, file_name)
            pdf_bytes = None
            if HAS_REPORTLAB:
                try:
                    pdf_bytes = build_report_pdf(result, file_name)
                except Exception as pdf_err:
                    st.caption(f"PDF generation failed: {pdf_err}")
            else:
                st.caption(
                    "PDF export needs the **reportlab** package. "
                    "Install with: `pip install reportlab` — Markdown download still works."
                )

            dl_col1, dl_col2, _ = st.columns([1, 1, 1.4])
            with dl_col1:
                st.download_button(
                    label="⬇  Download PDF Report",
                    data=pdf_bytes if pdf_bytes else b"",
                    file_name=f"Cerebral_Report_{pretty(name).replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    disabled=pdf_bytes is None,
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    label="⬇  Download Markdown",
                    data=md_report,
                    file_name=f"Cerebral_Report_{pretty(name).replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with st.expander("Preview full Markdown report"):
                st.markdown(md_report)

# Stepper is filled last so it reflects the final state of this run.
render_stepper(uploaded is not None, st.session_state.result is not None)

# =========================================================
# FOOTER / DISCLAIMER
# =========================================================
st.markdown("---")
st.markdown(
    block(
        """
<div class="footer">
  <div class="tags">
    <span class="tag">VGG16</span><span class="tag">TENSORFLOW</span>
    <span class="tag">STREAMLIT</span><span class="tag">EDUCATIONAL</span>
  </div>
  <b>CEREBRAL AI</b> &nbsp;·&nbsp; BRAIN MRI CLASSIFICATION PROJECT<br>
  Educational demonstration only. This application is not a medical diagnostic
  device and must not be used to make healthcare decisions.<br>
  Consult a qualified medical professional for interpretation of MRI scans.
</div>
"""
    ),
    unsafe_allow_html=True,
)

# Jump back to the top once new results are ready.
if st.session_state.pop("scroll_top", False):
    scroll_to_top()