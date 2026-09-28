# Web-app migration validation — 2026-09-28

The web app now runs from this repository using `docker compose up -d --build web`.
The ROS image remains a separate deployment. Both import the shared filtering implementation in `railpath/` and neural inference in `railpath/`.

Checks completed:

- Built both Docker images; verified the ROS node imports the shared filter.
- Passed 78 deployment tests and 84 original research-workspace regression tests.
- Passed browser camera/rendering tests and layout checks at four viewport sizes.
- Compared `cloud_with_fake_obj`'s first scan against the original app: identical rail and centerline arrays, 16 detected points, and nearest distance 98.75025939941406 m.
- Used the browser to load the `roundT_doubleT` merged map, select and place a square box, edit XYZ and scale, confirm placement, export the full bag, and replay the output without browser errors.
- Verified source and exported message timestamps, topic names, types, and message counts match.
- Inspected saved map-placement and export-replay screenshots. The replay reported 12 synthetic returns and two removed background points on its first scan.
- Confirmed the final deployment serves all eight source bags on port 8000.

Local test artifacts (ignored by Git):

- `outputs/obstacles_20260928_172843_a474de/`
- `outputs/migration_map_placement.png`
- `outputs/migration_export_replay.png`
- `outputs/migration_browser_result.json`

The parent workspace uses compatibility links for app/model/helper imports and a forwarding Compose file. Original migrated source files were retained in `outputs/.migration_backup_20260928/`. Existing source bag files were hard-linked into `data/` to avoid duplicating disk usage; treat both locations as read-only. Existing exports and caches were moved into this repository's `outputs/`, with a compatibility link at the old location.
