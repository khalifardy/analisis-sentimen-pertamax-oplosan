"""Generate BAB IV (Hasil dan Pembahasan) Word document."""

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt


OUTPUT_PATH = "docs/BAB_IV_Hasil_dan_Pembahasan.docx"


def set_cell_shading(cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def add_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "6")
        border.set(qn("w:color"), "000000")
        borders.append(border)
    tbl_pr.append(borders)


def add_run(paragraph, text: str, *, bold: bool = False, italic: bool = False, size: int = 12) -> None:
    run = paragraph.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:cs"), "Times New Roman")


def add_paragraph(
    doc,
    text: str = "",
    *,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    first_line_indent_cm: float = 1.27,
    space_after_pt: int = 6,
    bold: bool = False,
    italic: bool = False,
    size: int = 12,
):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(space_after_pt)
    pf.line_spacing = 1.5
    if first_line_indent_cm:
        pf.first_line_indent = Cm(first_line_indent_cm)
    if text:
        add_run(p, text, bold=bold, italic=italic, size=size)
    return p


def add_heading(doc, text: str, level: int = 1) -> None:
    if level == 0:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(12)
        p.paragraph_format.space_before = Pt(12)
        add_run(p, text, bold=True, size=14)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Cm(0)
    add_run(p, text, bold=True, size=12)


def add_caption(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.first_line_indent = Cm(0)
    add_run(p, text, italic=True, size=11)


def add_table_caption(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Cm(0)
    add_run(p, text, bold=True, italic=True, size=11)


def add_runs(paragraph, segments) -> None:
    for seg in segments:
        text = seg["text"]
        if text is None:
            text = ""
        add_run(
            paragraph,
            text,
            bold=seg.get("bold", False),
            italic=seg.get("italic", False),
            size=seg.get("size", 12),
        )


def add_bullet(doc, segments, *, indent_cm: float = 1.27) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.left_indent = Cm(indent_cm + 0.5)
    pf.first_line_indent = Cm(-0.5)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.5
    add_runs(p, segments)


def main() -> None:
    doc = Document()

    for section in doc.sections:
        section.left_margin = Cm(3.17)
        section.right_margin = Cm(3.17)
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    add_heading(doc, "BAB IV", level=0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    p.paragraph_format.first_line_indent = Cm(0)
    add_run(p, "HASIL DAN PEMBAHASAN", bold=True, size=14)

    add_heading(doc, "A. Validitas Dataset dan Kesepakatan Inter-Annotator", level=1)
    p = add_paragraph(
        doc,
        (
            "Sebelum memasuki tahap klasifikasi, reliabilitas dari dataset yang dianotasi secara "
            "mandiri dievaluasi menggunakan koefisien Cohen\u2019s Kappa (\u03ba) untuk mengukur "
            "tingkat kesepakatan antara dua annotator independen. Proses anotasi data mentah ini "
            "menghasilkan skor kesepakatan awal sebesar 0,84, yang berdasarkan kriteria Landis dan "
            "Koch diklasifikasikan ke dalam tingkat reliabilitas \u201cSangat Baik\u201d atau "
            "\u201cHampir Sempurna\u201d. Untuk menghilangkan bias residu, baris data yang memiliki "
            "perbedaan label opini antar-annotator disaring secara sistematis, sehingga menghasilkan "
            "satu kesatuan Dataset Kesepakatan yang memiliki validitas tinggi untuk digunakan pada "
            "fase eksperimen kodingan model."
        ),
    )

    add_heading(doc, "B. Perbandingan Performa Klasifikasi Secara Menyeluruh", level=1)
    add_paragraph(
        doc,
        (
            "Performa dari beberapa arsitektur machine learning dan deep learning diuji secara "
            "komparatif menggunakan fitur token teks yang telah melalui proses stemming Sastrawi. "
            "Eksperimen ini dikombinasikan dengan berbagai teknik penyeimbangan data "
            "(data-balancing) untuk mengatasi ketimpangan kelas. Tabel I di bawah ini merangkum "
            "metrik evaluasi yang berfokus pada nilai akurasi dan macro-averaged F1-score."
        ),
    )

    add_table_caption(
        doc,
        "TABEL I: Matriks Evaluasi Performa Model Klasifikasi Sastrawi",
    )

    headers = ["Arsitektur Model", "Teknik Balancing", "Akurasi (%)", "Macro F1-Score"]
    rows = [
        ["Random Forest", "ROS", "85,12%", "0,8418"],
        ["SVM", "ROS", "85,12%", "0,8415"],
        ["MLP Baseline", "ROS", "85,05%", "0,8418"],
        ["Naive Bayes", "ROS", "78,43%", "0,7650"],
        ["MLP Keras Tuner", "RUS", "24,88%", "0,1328"],
    ]

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    add_borders(table)

    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.text = ""
        p_cell = cell.paragraphs[0]
        p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run(p_cell, header, bold=True, size=11)
        set_cell_shading(cell, "D9E2F3")

    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.first_line_indent = Cm(0)
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.text = ""
            p_cell = cell.paragraphs[0]
            p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_run(p_cell, value, size=11)

    add_paragraph(
        doc,
        (
            "Berdasarkan data pada Tabel I, algoritma Random Forest dan Support Vector Machine (SVM) "
            "yang dioptimalkan dengan teknik Random Over-Sampling (ROS) berhasil mencatatkan performa "
            "puncak dengan nilai akurasi tertinggi sebesar 85,12%. Pencapaian performa ini terbukti "
            "sukses melampaui batas ambang kesepakatan inter-annotator awal "
            "(85,12% > 84%). Hal tersebut secara ilmiah memvalidasi bahwa reduksi variasi morfologi "
            "kata melalui stemming Sastrawi efektif membantu model dalam memetakan bobot fitur "
            "TF-IDF secara lebih konsisten."
        ),
    )

    add_heading(doc, "C. Analisis Komparatif Gambar Confusion Matrix", level=1)
    add_paragraph(
        doc,
        (
            "Meskipun Random Forest dan SVM menghasilkan angka akurasi global yang identik (85,12%), "
            "visualisasi matriks kontingensi pada grafik Confusion Matrix milik kedua model "
            "menunjukkan karakteristik prediksi internal yang sangat bertolak belakang:"
        ),
        first_line_indent_cm=1.27,
    )

    add_bullet(
        doc,
        [
            {"text": "Model Random Forest + ROS: ", "bold": True},
            {
                "text": (
                    "Model ini menunjukkan akurasi tebakan yang sangat dominan pada kelas mayoritas, "
                    "dengan berhasil mengklasifikasikan 89,27% (4.276 sampel) data sentimen Negatif "
                    "secara tepat. Kendati demikian, model ini mengalami bias ekstrem pada kelas "
                    "minoritas, di mana ia hanya mampu menebak 23,81% (25 sampel) pada sentimen "
                    "Positif, dan justru sering terkecoh memprediksinya sebagai sentimen Negatif "
                    "(58,10%)."
                )
            },
        ],
    )
    add_bullet(
        doc,
        [
            {"text": "Model SVM + ROS: ", "bold": True},
            {
                "text": (
                    "Sebaliknya, SVM berhasil membangun batas keputusan (hyperplane) yang jauh lebih "
                    "kokoh dan adil. SVM secara signifikan mampu menekan degradasi prediksi kelas "
                    "minoritas dengan meraih true positive rate sebesar 48,57% (51 sampel) untuk "
                    "sentimen Positif, sembari tetap mempertahankan kestabilan akurasi pada kelas "
                    "Netral (71,88%) dan kelas Negatif (79,90%)."
                )
            },
        ],
    )

    add_heading(
        doc,
        "D. Evaluasi Daya Diskriminasi Model via Analisis Kurva ROC",
        level=1,
    )
    add_paragraph(
        doc,
        (
            "Kapasitas model dalam memisahkan probabilitas tiap kelas sentimen diuji lebih lanjut "
            "menggunakan grafik kurva Receiver Operating Characteristic (ROC) dan perolehan nilai "
            "AUC (Area Under Curve):"
        ),
    )

    add_bullet(
        doc,
        [
            {"text": "Kurva ROC Random Forest: ", "bold": True},
            {
                "text": (
                    "Model ini menghasilkan daya diskriminasi yang fluktuatif antar kelas, di mana "
                    "kemampuan pemisahan tertingginya berada pada kelas Netral (AUC = 0,880), "
                    "namun mengalami penurunan pada kelas Positif (AUC = 0,826)."
                )
            },
        ],
    )
    add_bullet(
        doc,
        [
            {"text": "Kurva ROC SVM: ", "bold": True},
            {
                "text": (
                    "Berbeda dengan Random Forest, model SVM memamerkan kemampuan pemisahan kelas "
                    "yang sangat stabil dan merata di seluruh dimensi semantik, yaitu kelas Negatif "
                    "(AUC = 0,846), kelas Netral (AUC = 0,846), dan kelas Positif (AUC = 0,856)."
                )
            },
        ],
    )

    add_paragraph(
        doc,
        (
            "Fakta bahwa SVM menghasilkan nilai AUC tertinggi justru pada kelas minoritas Positif "
            "(0,856) membuktikan keunggulan generalisasi matematisnya dalam menangani distribusi "
            "data teks Twitter yang timpang. Dengan demikian, SVM + ROS menjadi model yang paling "
            "direkomendasikan untuk diimplementasikan pada tugas analisis sentimen ini."
        ),
    )

    doc.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
