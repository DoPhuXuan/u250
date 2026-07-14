#!/usr/bin/env python3
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


FIELDS = [
    "top",
    "status",
    "latency_min",
    "latency_avg",
    "latency_max",
    "interval_min",
    "interval_max",
    "bram_18k",
    "bram_18k_avail",
    "bram_18k_pct",
    "dsp",
    "dsp_avail",
    "dsp_pct",
    "ff",
    "ff_avail",
    "ff_pct",
    "lut",
    "lut_avail",
    "lut_pct",
    "uram",
    "uram_avail",
    "uram_pct",
    "clock_target_ns",
    "clock_estimated_ns",
    "report_dir",
]


def local_name(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def xml_text(root, path, default=""):
    node = root
    for want in path:
        found = None
        for child in list(node):
            if local_name(child.tag) == want:
                found = child
                break
        if found is None:
            return default
        node = found
    return (node.text or "").strip()


def pct(used, avail):
    try:
        used_f = float(str(used).replace(",", ""))
        avail_f = float(str(avail).replace(",", ""))
        if avail_f <= 0:
            return ""
        return f"{100.0 * used_f / avail_f:.2f}"
    except ValueError:
        return ""


def empty_row(top, report_dir, status):
    row = {field: "" for field in FIELDS}
    row.update({
        "top": top,
        "status": status,
        "report_dir": str(report_dir),
    })
    return row


def parse_xml(xml_path, report_dir, fallback_top):
    root = ET.parse(xml_path).getroot()
    top = xml_text(root, ["UserAssignments", "TopModelName"], fallback_top)
    row = empty_row(top, report_dir, "ok")

    row["clock_target_ns"] = xml_text(root, ["UserAssignments", "TargetClockPeriod"])
    row["clock_estimated_ns"] = xml_text(root, ["PerformanceEstimates", "SummaryOfTimingAnalysis", "EstimatedClockPeriod"])
    row["latency_min"] = xml_text(root, ["PerformanceEstimates", "SummaryOfOverallLatency", "Best-caseLatency"])
    row["latency_avg"] = xml_text(root, ["PerformanceEstimates", "SummaryOfOverallLatency", "Average-caseLatency"])
    row["latency_max"] = xml_text(root, ["PerformanceEstimates", "SummaryOfOverallLatency", "Worst-caseLatency"])
    row["interval_min"] = xml_text(root, ["PerformanceEstimates", "SummaryOfOverallLatency", "Interval-min"])
    row["interval_max"] = xml_text(root, ["PerformanceEstimates", "SummaryOfOverallLatency", "Interval-max"])

    for out_name, xml_name in {
        "bram_18k": "BRAM_18K",
        "dsp": "DSP",
        "ff": "FF",
        "lut": "LUT",
        "uram": "URAM",
    }.items():
        used = xml_text(root, ["AreaEstimates", "Resources", xml_name])
        avail = xml_text(root, ["AreaEstimates", "AvailableResources", xml_name])
        row[out_name] = used
        row[f"{out_name}_avail"] = avail
        row[f"{out_name}_pct"] = pct(used, avail)

    return row


def used_and_pct(text):
    match = re.match(r"\s*([\d,.-]+)(?:\s*\(([^%]+)%\))?", text)
    if not match:
        return "", ""
    used = match.group(1).replace(",", "")
    pct_text = match.group(2) or ""
    return used, pct_text


def parse_rpt(rpt_path, report_dir):
    text = rpt_path.read_text(encoding="utf-8", errors="ignore")
    top_match = re.search(r"Synthesis Summary Report of '([^']+)'", text)
    top = top_match.group(1) if top_match else rpt_path.stem
    row = empty_row(top, report_dir, "ok_rpt_only")

    for line in text.splitlines():
        if f"|+ {top}" not in line:
            continue
        cols = [col.strip() for col in line.split("|")]
        if len(cols) < 15:
            continue

        row["latency_min"] = cols[4]
        row["latency_avg"] = cols[4]
        row["latency_max"] = cols[4]
        row["interval_min"] = cols[7]
        row["interval_max"] = cols[7]

        for out_name, col_idx in {
            "bram_18k": 10,
            "dsp": 11,
            "ff": 12,
            "lut": 13,
            "uram": 14,
        }.items():
            used, pct_text = used_and_pct(cols[col_idx])
            row[out_name] = "" if used == "-" else used
            row[f"{out_name}_pct"] = pct_text
        break

    return row


def find_reports(root):
    rows = []
    standalone_rows = []
    current_dir = root / "reports" / "current"
    for standalone_rpt in sorted(current_dir.glob("csynth*.rpt")):
        standalone_rows.append(parse_rpt(standalone_rpt, current_dir))

    project_rows = []
    project_root = root / "build" / "hls"
    for project in sorted(project_root.glob("proj_*")):
        top = project.name.replace("proj_", "", 1)
        project_rpt = project / "csynth.rpt"
        if project_rpt.exists():
            project_rows.append(parse_rpt(project_rpt, project))
            continue

        report_dir = project / "solution1" / "syn" / "report"
        if not report_dir.exists():
            project_rows.append(empty_row(top, report_dir, "missing_report_dir"))
            continue

        exact_xml = report_dir / f"{top}_csynth.xml"
        xml_files = sorted(report_dir.glob("*_csynth.xml"))
        if exact_xml.exists():
            project_rows.append(parse_xml(exact_xml, report_dir, top))
        elif xml_files:
            row = parse_xml(xml_files[0], report_dir, top)
            row["top"] = top
            row["status"] = "fallback_xml_check_name"
            project_rows.append(row)
        else:
            project_rows.append(empty_row(top, report_dir, "missing_csynth_xml"))

    seen = {row.get("top") for row in project_rows}
    for standalone_row in standalone_rows:
        top = standalone_row.get("top")
        if top not in seen:
            rows.append(standalone_row)
            seen.add(top)
    rows.extend(project_rows)
    return rows


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    cols = ["top", "status", "latency_max", "interval_max", "dsp", "dsp_pct", "bram_18k", "bram_18k_pct", "uram", "uram_pct", "lut", "ff", "clock_estimated_ns"]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |\n")


def print_rows(rows):
    cols = ["top", "status", "latency_max", "interval_max", "dsp", "bram_18k", "uram", "lut", "ff", "clock_estimated_ns"]
    if not rows:
        print("no reports found")
        return
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in cols}
    print("  ".join(col.ljust(widths[col]) for col in cols))
    print("  ".join("-" * widths[col] for col in cols))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    current_dir = root / "reports" / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else current_dir / "ffn_kernel_report_summary.csv"
    md_path = csv_path.with_suffix(".md")

    rows = find_reports(root)
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print_rows(rows)
    print(f"\nwrote {len(rows)} rows")
    print(f"csv: {csv_path}")
    print(f"md : {md_path}")


if __name__ == "__main__":
    main()
