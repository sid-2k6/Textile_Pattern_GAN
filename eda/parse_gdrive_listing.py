#!/usr/bin/env python3
"""
Parse a `gdown --folder` listing into a per-folder file-count table.

Why this exists
---------------
The dataset lives in a public Google Drive folder. `gdown --folder` is the
easiest way to enumerate it, BUT for anonymous access Google Drive only returns
the first *page* (~50 items) of each folder, so gdown's own counts are capped at
50 files/folder. The dataset authors worked around this by embedding the true
count in the folder name (e.g. "all_motifs (864)"), which this parser surfaces.

Usage
-----
    gdown --folder "<share-url>" --remaining-ok -O /tmp/dl > listing.txt 2>&1
    python parse_gdrive_listing.py listing.txt
"""
import re, sys, json, collections

folder_re = re.compile(r"^Retrieving folder ([A-Za-z0-9_-]+) (.*)$")
file_re = re.compile(r"^Processing file ([A-Za-z0-9_-]+) (.*)$")
name_count_re = re.compile(r"\((\d+)\)\s*$")


def parse(path):
    folders, files_by_folder, current = [], collections.OrderedDict(), None
    for line in open(path):
        line = line.rstrip("\n")
        m = folder_re.match(line)
        if m:
            fid, name = m.group(1), m.group(2)
            if name in ("contents", "completed"):
                continue
            current = fid
            folders.append((fid, name))
            files_by_folder.setdefault(fid, [])
            continue
        m = file_re.match(line)
        if m and current is not None:
            files_by_folder[current].append((m.group(1), m.group(2)))
    return folders, files_by_folder


if __name__ == "__main__":
    listing = sys.argv[1] if len(sys.argv) > 1 else "listing.txt"
    folders, fbf = parse(listing)
    print(f"folders listed: {len(folders)}   files listed: "
          f"{sum(len(v) for v in fbf.values())} (capped at 50/folder)\n")
    for fid, name in folders:
        listed = len(fbf.get(fid, []))
        m = name_count_re.search(name)
        true = m.group(1) if m else ("=listed" if listed < 50 else ">=50")
        print(f"  listed={listed:4d}  true={true:>7}  {name}")
    json.dump({"folders": folders, "files_by_folder": fbf},
              open("listing_parsed.json", "w"))
