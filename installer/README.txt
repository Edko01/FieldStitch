FIELDSTITCH 1.2  -  offline orthomosaics from drone photos
=========================================================

Builds a georeferenced orthomosaic (GeoTIFF) from a folder of nadir mapping
photos, then lets you split it into fields, grid tiles or imported
boundaries and export one GeoTIFF per area. Everything runs on this PC;
no internet connection is needed.

Built and tuned for the DJI Matrice 4E (RTK, mechanical shutter), and it
also works with other DJI drones and most cameras that record GPS.


STARTING
--------
Open FieldStitch from the Start menu or the desktop icon. It opens in its
own window. Close the window when you're done; if a job is still
processing, it keeps running in the background and FieldStitch shows its
progress again the next time you open it.

The first time you run the installer, Windows SmartScreen may say it
"protected your PC" because the installer isn't code-signed. Click
"More info" -> "Run anyway".


WORKFLOW
--------
1  Photos   Create a project, then add one or more photo folders (an SD
            card's DCIM folder works; subfolders are included).
            The map shows where each photo was taken:
              green = RTK fixed, yellow = standard GPS,
              red ring = tilted photo (skipped automatically).
            "Pick photos on map" lets you process only part of a big job.

2  Process  Choose a quality:
              Fast      half-resolution output, quickest; scouting.
              Balanced  full-resolution output (default).
              Accurate  finer feature matching, about 2x slower.
            Output options:
              Surface "Fields" (smooth ground, best for crops) or
              "Trees & buildings" (follows taller objects more closely).
              File type: Compact (JPEG-compressed GeoTIFF, about 10x
              smaller) or Lossless.
            Steps that already finished are reused if you cancel and
            restart, so changing output options re-runs quickly.

3  Areas    Draw rectangles or polygons, split into a grid (by tile size
            or rows x columns), or import field boundaries from
            KML / KMZ / Shapefile (.shp with its .prj) / GeoJSON.
            Tick the areas you want and click Export. Each area becomes
            its own GeoTIFF, clipped to the shape, with transparent
            outside. "Save area outlines as KML" writes the shapes too.


OUTPUT FILES  (in the project's "outputs" folder)
-------------------------------------------------
  <name>_orthomosaic.tif   the full mosaic, GeoTIFF, WGS84 / UTM zone,
                           tiled with overviews (opens fast in QGIS, ArcGIS,
                           Global Mapper, SMS, etc.)
  <name>_surface.tif       coarse elevation surface used for the mosaic
  report.json              processing statistics
  preview.png              small preview image
Area exports go to the project's "exports" folder unless you choose
another one.


GETTING THE BEST RESULT
-----------------------
  * Fly a grid with about 75% front / 70% side overlap at a constant
    height. Use RTK fixed when you can: it is what makes the mosaic land
    on the correct spot on the ground, with no ground control needed.
  * Keep the gimbal straight down (-90). Tilted photos are skipped.
  * Zoom/tele photos mixed into the folder are ignored when they are a
    minority; for best results keep them in a separate folder.
  * The report's "Seam alignment" shows how well neighbouring photos line
    up. Over trees or buildings it includes normal 3D lean.


SPEED AND MEMORY (rough guide, 8-core laptop, 16-32 GB RAM)
-----------------------------------------------------------
  Matrice 4E 20 MP photos, Balanced:
     300 photos  ~ 3-5 min      1,000 photos ~ 10-15 min
  Fast is roughly 40% quicker. 16 GB RAM is the practical minimum for
  1,000+ photo jobs. The project's "work" folder holds cached
  intermediate data; delete it after you are done to free space.


UPDATES
-------
  Click "Check for updates" (top right) while the PC is online.
  If a newer version exists you'll see what changed; click "Install
  and restart". FieldStitch downloads the update, swaps it in and
  reopens in a few seconds. Your projects and exports are not touched.
  FieldStitch also checks quietly each time it opens; a yellow dot on
  the button means an update is waiting. Offline, nothing happens.
  Updates can't be installed while processing or exporting is running.


TROUBLESHOOTING
---------------
  * "Lost contact with FieldStitch": FieldStitch was closed. Open it again
    from the Start menu; projects are saved automatically.
  * Browse... opens the folder picker in a separate window; it may
    appear behind the browser (check the taskbar). You can also paste
    a folder path into the box.
  * If processing stops with an error, click "Show log". The full log
    is in the project folder as log.txt.

To uninstall, use Settings -> Apps -> Installed apps -> FieldStitch.
Your projects in Documents\FieldStitch Projects are kept. A program log
is in %LOCALAPPDATA%\FieldStitch\logs if something goes wrong.
