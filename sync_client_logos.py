"""One-time script: map existing bucket logos to clients, scrape missing ones, update logo_url."""

import re
import time
from urllib.parse import urlparse

import requests
from supabase import create_client

from prospect_outreach.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

BUCKET = "client-logos"
BASE_PUBLIC_URL = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}"


def _slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s).strip("-")
    s = re.sub(r"-+", "-", s)
    return s


def _extract_domain(url: str) -> str | None:
    if not url or not url.strip():
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        domain = parsed.hostname
        if domain and domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def _fetch_logo(domain: str) -> tuple[bytes, str] | None:
    # Try Clearbit
    try:
        resp = requests.get(f"https://logo.clearbit.com/{domain}", timeout=10)
        if resp.status_code == 200 and len(resp.content) > 100:
            ct = resp.headers.get("content-type", "image/png")
            return resp.content, ct
    except Exception:
        pass

    # Fallback: Google Favicon
    try:
        resp = requests.get(
            f"https://www.google.com/s2/favicons?domain={domain}&sz=128", timeout=10
        )
        if resp.status_code == 200 and len(resp.content) > 100:
            ct = resp.headers.get("content-type", "image/png")
            return resp.content, ct
    except Exception:
        pass

    return None


def _get_extension(content_type: str) -> str:
    ct = content_type.lower()
    if "svg" in ct:
        return ".svg"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "ico" in ct:
        return ".ico"
    return ".png"


def main():
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Fetch all clients
    clients = sb.table("client_referencing_data").select("id, client_name, client_url, logo_url").execute().data
    print(f"Clients in DB: {len(clients)}")

    # Fetch existing bucket files
    bucket_files = sb.storage.from_(BUCKET).list()
    bucket_names = {f["name"] for f in bucket_files}
    print(f"Files in bucket: {len(bucket_names)}")

    # Build manual mapping for known mismatches between client_name slugs and existing filenames
    # (existing bucket uses abbreviated names from client_references table)
    manual_map = {
        "Alsco Uniforms": "alsco.png",
        "BCDS Group": "bcds.png",
        "Brickworks (Austral Bricks)": "brickworks.png",
        "Caritas Australia": "caritas.png",
        "Cellulant Group": "cellulant.png",
        "Chill Logistics": "chill.png",
        "Clear K12, Inc": "clear-k12-inc.png",
        "CompareClub": "compare-club.png",
        "Construction Forms, Inc. (CFI)": "cfi.png",
        "Cornerstone Health": "corner-stone.png",
        "Empist": "deskware-empist.png",
        "Flex Equip by MND of NSW": "flex-equip-mnd.png",
        "Glen-Gery": "glen-gerry.png",
        "Guardian Avionics": "guardianavionics.png",
        "Gulf International Bank": "gib.png",
        "Integrated Financial Technologies": "ift.png",
        "Motor Nuerone Disease of NSW": "mnd.png",
        "NewPort & Oclaro": "newport-laser-oclaro.png",
        "Panama Telemetro": "telemetro.png",
        "Partners in Performance (Now Accenture)": "partners-in-performance-now-accenture.png",
        "Raise Foundation": "raise-foundation.png",
        "Regents Capital": "regents.png",
        "Remunerator Australia": "remunerator.png",
        "Schlumberger": "actaris-ex-schlumberger-aquired-by-itron.png",
        "SmartGroup Corporation": "smart-group.png",
        "SwiftLoans": "swift-loans.png",
        "TFG Financial": "tfg.png",
        "Total Processing (Nomupay)": "total-processing-nomupay.png",
        "Zenudy (Staredla)": "zenudy-staredla.png",
    }

    used_files = set()
    success = 0
    failed = 0
    skipped = 0
    failed_clients = []

    for i, client in enumerate(sorted(clients, key=lambda c: c["client_name"]), 1):
        cid = client["id"]
        name = client["client_name"]
        url = client["client_url"]
        prefix = f"[{i}/{len(clients)}] {name:45s}"

        # Skip clients that already have a logo
        if client.get("logo_url"):
            print(f"{prefix} -> SKIPPED (logo already set)")
            skipped += 1
            continue

        # Check manual map first
        if name in manual_map:
            filename = manual_map[name]
            if filename in bucket_names:
                logo_url = f"{BASE_PUBLIC_URL}/{filename}"
                sb.table("client_referencing_data").update({"logo_url": logo_url}).eq("id", cid).execute()
                used_files.add(filename)
                success += 1
                print(f"{prefix} -> MAPPED to existing: {filename}")
                continue

        # Check if slug matches an existing file
        slug = _slugify(name)
        for ext in [".png", ".svg", ".jpg", ".ico"]:
            candidate = slug + ext
            if candidate in bucket_names:
                logo_url = f"{BASE_PUBLIC_URL}/{candidate}"
                sb.table("client_referencing_data").update({"logo_url": logo_url}).eq("id", cid).execute()
                used_files.add(candidate)
                success += 1
                print(f"{prefix} -> MATCHED slug: {candidate}")
                break
        else:
            # No existing file — scrape
            domain = _extract_domain(url)
            if not domain:
                print(f"{prefix} -> SKIPPED (no valid URL)")
                skipped += 1
                continue

            time.sleep(0.3)
            result = _fetch_logo(domain)
            if not result:
                print(f"{prefix} -> FAILED (could not fetch logo for {domain})")
                failed += 1
                failed_clients.append(name)
                continue

            image_bytes, content_type = result
            ext = _get_extension(content_type)
            filename = slug + ext

            try:
                sb.storage.from_(BUCKET).upload(
                    filename, image_bytes,
                    {"content-type": content_type, "upsert": "true"},
                )
                logo_url = f"{BASE_PUBLIC_URL}/{filename}"
                sb.table("client_referencing_data").update({"logo_url": logo_url}).eq("id", cid).execute()
                used_files.add(filename)
                success += 1
                print(f"{prefix} -> SCRAPED & uploaded: {filename}")
            except Exception as e:
                print(f"{prefix} -> UPLOAD FAILED: {e}")
                failed += 1
                failed_clients.append(name)

    # Summary
    print("\n" + "=" * 70)
    print(f"DONE  |  Success: {success}  |  Failed: {failed}  |  Skipped: {skipped}")
    print(f"Total files now used: {len(used_files)}  |  Clients: {len(clients)}")

    if failed_clients:
        print(f"\nFailed clients:")
        for c in failed_clients:
            print(f"  - {c}")

    # Files in bucket that were never used
    unused = bucket_names - used_files
    if unused:
        print(f"\nUnused files in bucket (safe to delete, {len(unused)} files):")
        for f in sorted(unused):
            print(f"  - {f}")
    else:
        print("\nNo unused files in bucket.")


if __name__ == "__main__":
    main()
