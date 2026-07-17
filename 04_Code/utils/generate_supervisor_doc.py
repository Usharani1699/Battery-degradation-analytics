"""
Generate supervisor weekly update PDF for EG7030 Dissertation.
Output: 06_Draft_Update/Supervisor_Updates/Supervisor_Update_Week5_17Jul2026.pdf
Usage:  python 04_Code/utils/generate_supervisor_doc.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "06_Draft_Update" / "Supervisor_Updates"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT     = OUT_DIR / "Supervisor_Update_Week5_17Jul2026.pdf"

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1B2B4A")
AMBER  = colors.HexColor("#D4890A")
SAGE   = colors.HexColor("#3E7D52")
LIGHT  = colors.HexColor("#F4F6FA")
MUTED  = colors.HexColor("#555E6E")
WHITE  = colors.white

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

H1 = style("H1", "Heading1",
           fontSize=18, textColor=NAVY, spaceAfter=4,
           fontName="Helvetica-Bold")
H2 = style("H2", "Heading2",
           fontSize=12, textColor=NAVY, spaceAfter=3, spaceBefore=10,
           fontName="Helvetica-Bold")
H3 = style("H3", "Heading3",
           fontSize=10, textColor=AMBER, spaceAfter=2, spaceBefore=6,
           fontName="Helvetica-Bold")
BODY = style("BODY",
             fontSize=10, leading=15, textColor=colors.black,
             spaceAfter=4, alignment=TA_JUSTIFY,
             fontName="Helvetica")
BULLET = style("BULLET",
               fontSize=10, leading=14, textColor=colors.black,
               leftIndent=16, spaceAfter=3,
               fontName="Helvetica")
SMALL = style("SMALL",
              fontSize=8, leading=11, textColor=MUTED,
              fontName="Helvetica")
META = style("META",
             fontSize=9, leading=13, textColor=MUTED,
             fontName="Helvetica")
QUESTION = style("QUESTION",
                 fontSize=10, leading=14, textColor=NAVY,
                 leftIndent=12, spaceAfter=3,
                 fontName="Helvetica-Oblique")
HIGHLIGHT = style("HIGHLIGHT",
                  fontSize=10, leading=14, textColor=SAGE,
                  fontName="Helvetica-Bold")

def bullet(text):
    return Paragraph(f"&#8226;&#160; {text}", BULLET)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=AMBER, spaceAfter=8, spaceBefore=2)

def section(title):
    return [Paragraph(title, H2), hr()]

# ── Document ──────────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title="Supervisor Weekly Update — Week 5",
        author="Usha Rani Vamanagiri"
    )

    story = []

    # ── Header block ──────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("<b>EG7030 Dissertation</b><br/>Supervisor Weekly Update — Week 5", H1),
        Paragraph(
            "Student:&#160; Usha Rani Vamanagiri (u2965962)<br/>"
            "Programme:&#160; MSc Electric Vehicles and Energy Storage<br/>"
            "Date:&#160; 17 July 2026<br/>"
            "Module:&#160; EG7030 — Dissertation",
            META)
    ]]
    header_tbl = Table(header_data, colWidths=[10*cm, 6*cm])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",   (0, 0), (-1, -1), LIGHT),
        ("ROWPADDING",   (0, 0), (-1, -1), 10),
        ("LINEBELOW",    (0, 0), (-1, -1), 1.5, NAVY),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── 1. Research question ──────────────────────────────────────────────────
    story += section("1. Research Question")
    story.append(Paragraph(
        "This dissertation proposes, validates, and evaluates the <b>Fleet Stress Index (FSI)</b> — "
        "a composite, physics-grounded feature index designed to bridge the gap between "
        "laboratory battery datasets (constant-current cycling, KI = 0) and real-world EV fleet "
        "operation (highly variable current profiles, KI = 0.60–0.81). The central question is "
        "whether FSI can improve SoH prediction accuracy and provide early degradation warning "
        "for commercial EV fleets.", BODY))

    # ── 2. Work completed this week ───────────────────────────────────────────
    story += section("2. Work Completed This Week")

    story.append(Paragraph("<b>FSI Validation Pipeline — All 5 Datasets Complete</b>", H3))
    results_data = [
        ["Dataset", "Chemistry", "Key Result", "Status"],
        ["CALCE (UMD)", "LiCoO₂", "RMSE 3.73%, R² = 0.9839 (XGBoost)", "Training"],
        ["NASA PCoE RW", "LiCoO₂", "Cross-profile validation passed", "Validated"],
        ["Oxford LFP", "LFP", "Cross-chemistry validation passed", "Validated"],
        ["NREL BLAST-Lite", "NMC/LFP/NCA", "RMSE 20.4%; 84% is calibration bias", "Validated"],
        ["Severson 2019", "LFP", "Spearman(FSI, −cycle life) ρ = 0.81, p < 0.001", "Validated"],
    ]
    rtbl = Table(results_data, colWidths=[3.8*cm, 2.8*cm, 7.0*cm, 2.4*cm])
    rtbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("LEADING",      (0, 0), (-1, -1), 12),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR",    (4, 1), (4, 5), SAGE),
    ]))
    story.append(rtbl)
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("<b>Fleet DNA KI Validation</b>", H3))
    ki_data = [
        ["Vehicle Class", "Trips", "KI Mean", "KI Std"],
        ["Delivery Trucks", "553", "0.604", "0.076"],
        ["Transit Buses", "472", "0.631", "0.084"],
        ["Refuse Trucks", "387", "0.813", "0.118"],
        ["CC Lab Baseline", "—", "0.000", "0.000"],
    ]
    ktbl = Table(ki_data, colWidths=[5*cm, 2.5*cm, 3*cm, 3*cm])
    ktbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("LEADING",      (0, 0), (-1, -1), 12),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",     (0, 4), (-1, 4), "Helvetica-Oblique"),
        ("TEXTCOLOR",    (0, 4), (-1, 4), MUTED),
    ]))
    story.append(ktbl)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Wilcoxon signed-rank tests confirm KI >> 0 for all vehicle classes (p < 0.001). "
        "This validates the core premise: lab CC cycling (KI = 0) cannot replicate the "
        "stress profile of real fleet operation.", BODY))

    story.append(Paragraph("<b>Project Restructure and Documentation</b>", H3))
    for b in [
        "Restructured project into 7 numbered academic reproducibility folders (Raw &#8594; Code &#8594; Processed &#8594; Results &#8594; Dissertation)",
        "Created <b>DATA_DICTIONARY.csv</b> — complete column definitions, FSI formula derivation, and dataset provenance for all 5 datasets",
        "Submitted 1,500-word progress report (weekly update form, 17 July 2026)",
    ]:
        story.append(bullet(b))
    story.append(Spacer(1, 0.2*cm))

    # ── 3. Key findings ───────────────────────────────────────────────────────
    story += section("3. Key Findings")
    findings = [
        ("<b>FSI correctly ranks degradation severity</b> across 23 distinct fast-charging "
         "protocols (Severson LFP): Spearman &#961; = 0.81, p &lt; 0.001. Cells on highest-KI "
         "protocols degrade 62% faster than single-step protocols."),
        ("<b>KI = 0 in all lab CC data; KI = 0.60–0.81 in all real fleet data</b> — confirming "
         "the training-deployment gap that FSI is designed to close."),
        ("<b>Cross-chemistry performance gap is calibration, not structure.</b> 84% of the "
         "20.4% BLAST RMSE is a chemistry-specific offset removable by one-parameter linear "
         "rescaling. SHAP feature rank correlation CALCE vs BLAST: &#961; &gt; 0.80."),
        ("<b>XGBoost in-sample performance (CALCE):</b> RMSE = 3.73%, R&#178; = 0.9839. "
         "SHAP confirms KI as the dominant feature by importance margin."),
    ]
    for f in findings:
        story.append(bullet(f))
    story.append(Spacer(1, 0.2*cm))

    # ── 4. Question for supervisor ────────────────────────────────────────────
    story += section("4. Question for Supervisor Guidance")

    q_box_data = [[
        Paragraph(
            "<b>T<sub>norm</sub> Asymmetry — Scope or Fix?</b><br/><br/>"
            "The current thermal stress term is T<sub>norm</sub> = |T<sub>avg</sub> &#8722; 25| / 25, "
            "which is <i>symmetric</i> — it treats 10&#176;C and 40&#176;C identically (both give "
            "T<sub>norm</sub> = 0.60). This contradicts the Arrhenius degradation model, where "
            "elevated temperature accelerates degradation more than cold of the same magnitude "
            "from reference.<br/><br/>"
            "An asymmetric alternative would be:<br/>"
            "&#8226; T &gt; 25&#176;C: T<sub>norm</sub> = (T &#8722; 25) / 25 (positive, "
            "Arrhenius-aligned)<br/>"
            "&#8226; T &#8804; 25&#176;C: T<sub>norm</sub> = (25 &#8722; T) / 50 (half-weight "
            "for cold, since cold slows reactions rather than accelerating degradation)<br/><br/>"
            "<b>Should I implement this correction within the dissertation scope, or state it as "
            "a clearly-scoped limitation and recommend it as future work?</b> I am aware it would "
            "require re-running the full validation pipeline and recalibrating FSI weights.",
            BODY)
    ]]
    q_box = Table(q_box_data, colWidths=[15.5*cm])
    q_box.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#FFF8EE")),
        ("LINERIGHT",   (0, 0), (0, -1), 3, AMBER),
        ("ROWPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(q_box)
    story.append(Spacer(1, 0.3*cm))

    # ── 5. Plan for next week ─────────────────────────────────────────────────
    story += section("5. Plan for Next Week")
    next_steps = [
        "Write <b>Chapter 3 — Methodology</b>: FSI formula derivation, dataset extraction pipeline, ML training protocol",
        "Write <b>Chapter 4 — Results</b>: All five dataset validation results, cross-validation matrix, SHAP analysis",
        "Generate <b>publication-quality figures</b>: KI bar chart by vehicle class, FSI vs SoH scatter, SHAP beeswarm, protocol ranking chart",
        "Begin <b>Chapter 5 — Discussion</b>: calibration gap framing, T<sub>norm</sub> limitation (pending supervisor guidance above), fleet deployment implications",
    ]
    for ns in next_steps:
        story.append(bullet(ns))

    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=NAVY))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "All code, data, and validation results are version-controlled. "
        "The full 1,500-word progress report and DATA_DICTIONARY are available on request.",
        SMALL))

    doc.build(story)
    print(f"Generated: {OUT}")

if __name__ == "__main__":
    build()
