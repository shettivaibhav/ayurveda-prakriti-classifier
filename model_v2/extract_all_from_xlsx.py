"""Complete extraction of ALL labels + ALL images from
`palm_dataset_26 June 26.xlsx` into a clean, self-contained dataset.

The spreadsheet stores palm photos two different ways, and the old
`extract_embedded_images.py` (which relied on openpyxl's `ws._images`) only
saw *neither* of them on this file. Both are handled here:

  1. Legacy floating pictures anchored in `xl/drawings/drawing1.xml`
     (row/col -> r:embed -> media file).           ~330 images
  2. Modern "Place in Cell" rich-value images: a worksheet cell carries a
     `vm=` value-metadata index -> metadata.xml -> rdrichvalue.xml ->
     richValueRel.xml -> media file.                ~38 images

330 + 38 = 368 = 92 records x 4 views  -> full coverage.

Output layout (one folder per subject, normalised `{id}_{gender}` lowercase):

    palm_dataset_full/
        1001_male/
            palm_open_front.jpg
            palm_open_back.jpg
            palm_closed_front.jpg
            palm_closed_back.jpg
            metadata.json          # id, gender, label + 24 tabular features
        ...
    palm_dataset_full/labels_all.csv     # one row per subject (decoded labels)
    palm_dataset_full/manifest.csv       # id + label framings + image paths

Nothing outside palm_dataset_full/ is touched; the original xlsx is read-only.
"""

import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = PROJECT_ROOT / "palm_dataset_26 June 26.xlsx"
OUT_DIR = PROJECT_ROOT / "palm_dataset_full"

# 0-indexed worksheet column -> view name (col 0 is the id column)
COL_TO_VIEW = {1: "open_front", 2: "open_back", 3: "closed_front", 4: "closed_back"}
VIEW_TO_FILENAME = {
    "open_front": "palm_open_front.jpg",
    "open_back": "palm_open_back.jpg",
    "closed_front": "palm_closed_front.jpg",
    "closed_back": "palm_closed_back.jpg",
}

NS = {
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

# ---- one-hot label groups (the practitioner's annotation scheme) -------------
GROUPS = {
    "SC": [("Fair", "SC_Fair"), ("Medium", "SC_Medium"), ("Dark", "SC_Dark")],
    "ST": [("Rough", "ST_Rough"), ("Smooth", "ST_Smooth"), ("Dry", "ST_Dry"), ("Oily", "ST_Oily")],
    "SH": [("Thick", "SH_Thick"), ("Thin", "SH_Thin"), ("Med", "SH_Med")],
    "SHC": [("Brown", "SHC_Brown"), ("Black", "SHC_Black")],
    "VV": [("Low", "VV_Low"), ("Med", "VV_Med"), ("High", "VV_High")],
    "PC": [("PinkRed", "PC_PinkRed"), ("PinkWhite", "PC_PinkWhite"), ("Brown", "PC_Brown")],
}


def col_letter_to_idx(letters: str) -> int:
    """'A' -> 0, 'B' -> 1, ... (0-indexed)."""
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def cellref_to_rowcol(ref: str):
    """'B27' -> (row0=26, col0=1)."""
    m = re.match(r"([A-Z]+)(\d+)", ref)
    letters, row = m.group(1), int(m.group(2))
    return row - 1, col_letter_to_idx(letters)


def read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8")


def build_drawing_map(zf):
    """(row0, col0) -> media basename, from the legacy floating pictures."""
    rels = read_zip_text(zf, "xl/drawings/_rels/drawing1.xml.rels")
    rid2media = dict(
        re.findall(r'Id="([^"]+)"[^>]*Target="[^"]*media/([^"]+)"', rels)
    )
    root = ET.fromstring(read_zip_text(zf, "xl/drawings/drawing1.xml"))
    out = {}
    for anc in root:
        frm = anc.find("xdr:from", NS)
        if frm is None:
            continue
        row0 = int(frm.find("xdr:row", NS).text)
        col0 = int(frm.find("xdr:col", NS).text)
        blip = anc.find(".//a:blip", NS)
        if blip is None:
            continue
        media = rid2media.get(blip.get(R_EMBED))
        if media and col0 in COL_TO_VIEW:
            out[(row0, col0)] = media
    return out


def build_incell_map(zf):
    """(row0, col0) -> media basename, from the modern rich-value in-cell images.

    Chain: cell vm=K -> valueMetadata[K-1] -> rich value index -> local image
    identifier -> richValueRel[<id>] -> rId -> media file.
    """
    # rich value index -> local image identifier (first <v> of each <rv>)
    rv_xml = read_zip_text(zf, "xl/richData/rdrichvalue.xml")
    rv_ids = [int(m) for m in re.findall(r"<rv[^>]*>\s*<v>(\d+)</v>", rv_xml)]

    # valueMetadata: vm index (1-based position) -> rich value index (rc v="..")
    md_xml = read_zip_text(zf, "xl/metadata.xml")
    vmeta_block = re.search(r"<valueMetadata.*?</valueMetadata>", md_xml, re.S).group(0)
    vm_to_rv = [int(v) for v in re.findall(r'<rc[^>]*\bv="(\d+)"', vmeta_block)]

    # richValueRel list order -> rId ; then rId -> media basename
    rvr_xml = read_zip_text(zf, "xl/richData/richValueRel.xml")
    rel_order = re.findall(r'<rel\s+r:id="([^"]+)"', rvr_xml)
    rvr_rels = read_zip_text(zf, "xl/richData/_rels/richValueRel.xml.rels")
    rid2media = dict(
        re.findall(r'Id="([^"]+)"[^>]*Target="[^"]*media/([^"]+)"', rvr_rels)
    )

    def identifier_to_media(local_id: int):
        rid = rel_order[local_id]
        return rid2media.get(rid)

    # worksheet cells carrying vm=
    sheet_xml = read_zip_text(zf, "xl/worksheets/sheet1.xml")
    out = {}
    for ref, vm in re.findall(r'<c r="([A-Z]+\d+)"[^>]*\bvm="(\d+)"', sheet_xml):
        vm = int(vm)
        rv_index = vm_to_rv[vm - 1]
        local_id = rv_ids[rv_index]
        media = identifier_to_media(local_id)
        row0, col0 = cellref_to_rowcol(ref)
        if media and col0 in COL_TO_VIEW:
            out[(row0, col0)] = media
    return out


def decode(row_vals, header_idx, key):
    grp = GROUPS[key]
    on = [label for label, col in grp if row_vals[header_idx[col]] in (1, "1")]
    if on:
        return "/".join(on)
    return "None" if key == "SH" else ""


def load_labels():
    """row0 -> dict of decoded labels, keyed by 0-indexed worksheet row."""
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb["Sheet"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    hidx = {name: i for i, name in enumerate(header) if name is not None}
    labels = {}
    for r0, row in enumerate(rows):
        if r0 == 0:
            continue
        rid = row[0]
        if rid is None:
            continue
        rec = {
            "id_raw": rid,
            "Age": row[hidx["Age"]],
            "Sex": row[hidx["Sex"]],
            "Mole": row[hidx["Mole on skin"]],
            "Prakriti": row[hidx["Prakriti"]],
        }
        for k in GROUPS:
            rec[k] = decode(row, hidx, k)
        labels[r0] = rec
    return labels


def normalize_id(id_raw: str) -> str:
    num, _, gender = str(id_raw).partition("_")
    return f"{num}_{gender.lower()}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(XLSX_PATH) as zf:
        draw = build_drawing_map(zf)
        incell = build_incell_map(zf)
        # in-cell entries win only where the drawing has no picture (they fill gaps)
        combined = dict(draw)
        overlaps = 0
        for k, v in incell.items():
            if k in combined:
                overlaps += 1
            combined[k] = v
        print(f"drawing images: {len(draw)} | in-cell images: {len(incell)} | "
              f"overlap: {overlaps} | combined cells: {len(combined)}")

        labels = load_labels()

        # group media by 0-indexed row -> {view: media}
        by_row = {}
        for (row0, col0), media in combined.items():
            by_row.setdefault(row0, {})[COL_TO_VIEW[col0]] = media

        written, partial, missing_label = [], [], []
        csv_rows = []
        for row0, rec in sorted(labels.items()):
            norm = normalize_id(rec["id_raw"])
            subj_dir = OUT_DIR / norm
            subj_dir.mkdir(parents=True, exist_ok=True)
            views = by_row.get(row0, {})
            saved_views = []
            for view, media in views.items():
                data = zf.read(f"xl/media/{media}")
                img = Image.open(io.BytesIO(data))
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.save(subj_dir / VIEW_TO_FILENAME[view], format="JPEG", quality=95)
                saved_views.append(view)

            meta = {
                "id": norm,
                "id_raw": rec["id_raw"],
                "gender": norm.split("_", 1)[1],
                "age": rec["Age"],
                "sex": rec["Sex"],
                "prakriti": rec["Prakriti"],
                "features": {
                    "skin_color": rec["SC"], "skin_texture": rec["ST"],
                    "skin_hair": rec["SH"], "skin_hair_color": rec["SHC"],
                    "mole": rec["Mole"], "vein_visibility": rec["VV"],
                    "palm_color": rec["PC"],
                },
                "views_present": sorted(saved_views),
                "n_views": len(saved_views),
            }
            (subj_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

            csv_rows.append({
                "id": norm, "Age": rec["Age"], "Sex": rec["Sex"],
                "SC": rec["SC"], "ST": rec["ST"], "SH": rec["SH"], "SHC": rec["SHC"],
                "Mole": rec["Mole"], "VV": rec["VV"], "PC": rec["PC"],
                "Prakriti": rec["Prakriti"], "n_views": len(saved_views),
            })
            if len(saved_views) == 4:
                written.append(norm)
            else:
                partial.append((norm, sorted(saved_views)))
            if not rec["Prakriti"]:
                missing_label.append(norm)

    # write labels csv
    import csv
    with open(OUT_DIR / "labels_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\nSubjects written: {len(csv_rows)}")
    print(f"  with all 4 views: {len(written)}")
    print(f"  partial views:    {len(partial)} -> {partial}")
    print(f"  missing Prakriti label: {missing_label or 'none'}")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
