FIELDSTITCH 1.5  -  offline orthomosaics from drone photos
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
FieldStitch opens on the Projects page.

New project  Click New project, choose the flight's folder (subfolders are
           included; an SD card's DCIM folder works) or pick individual
           photos. The project is named after the folder; change the name
           or where it is saved before clicking Create project. Once the
           photos are loaded, the processing options open.
           Open a GeoTIFF on the Projects page opens an existing
           orthomosaic so you can split and export it without processing.
           The Projects button at the top returns to this page; the pencil
           next to the project name renames it.

Photos     Shows the photos in the project. The map shows where each was
           taken: green = RTK fixed, yellow = standard GPS, red ring =
           tilted, grey ring = another camera (both skipped).
           "Add more photos to this project" is for the same field only;
           start a new project for a different flight or field.
           "Pick photos on map" processes only part of a big job.

Process    Choose a processing type (Fast, Balanced or Accurate) and the
           output resolution with the slider (Full = the photos' own
           detail). The window shows the area, the orthomosaic's file
           size and how long processing should take; the time estimate
           learns this PC's speed after each job.
           Advanced: surface "Fields" or "Trees & buildings", mosaic
           storage Compact (JPEG) or Lossless, exposure balancing.
           Progress, time left and Stop processing are in the bar along
           the bottom. Finished steps are reused if you stop and restart.

Layers     Show or hide the orthomosaic (with opacity), areas and photo
           positions.

Areas      Draw rectangles or polygons, split into a grid (by tile size
           or rows x columns), or import field boundaries from
           KML / KMZ / Shapefile (.shp with its .prj) / GeoJSON.
           "Save outlines as KML" writes the shapes too.

Export     Export the whole orthomosaic and/or each area as its own
           GeoTIFF (clipped to the shape, transparent outside). Choose
           Compact (JPEG) or Lossless, 100 / 50 / 25 % resolution, and
           whether to include overviews. The size is shown before you
           start.

Projects   The grid button at the top left lists your projects with a
           preview, starts a new one, and sets where new projects are
           saved (Documents\FieldStitch Projects by default).


OUTPUT FILES  (in the project's "outputs" folder)
-------------------------------------------------
  <name>_orthomosaic.tif   the full mosaic, GeoTIFF, WGS84 / UTM zone, tiled
  <name>_orthomosaic.tif.ovr  its overviews (zoomed-out levels). Keep the two
                           files together: QGIS, ArcGIS, Global Mapper, SMS
                           etc. use the .ovr automatically to open fast. For a
                           single self-contained file, use Export > Whole
                           orthomosaic.
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


SPEED AND MEMORY
----------------
  FieldStitch uses all CPU cores, and an NVIDIA graphics card (GeForce RTX,
  Quadro/RTX A-series) when there is one: feature detection runs on the
  card while the CPU decodes photos. Other PCs process on the CPU only.
  The Processing options window shows the expected time for the job on
  this PC, and the estimate learns the PC's speed after each job.
  16 GB RAM is the practical minimum for
  1,000+ photo jobs; 32 GB is better. The project's "work" folder holds
  cached intermediate data; delete it after you are done to free space.
  Advanced > "Use the NVIDIA graphics card" turns the GPU off if needed.


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
  * The folder and file pickers open in a separate window; it may
    appear behind FieldStitch (check the taskbar). You can also paste
    a folder path into the Import panel.
  * If processing stops with an error, click "Log" in the bottom bar.
    The full log is in the project folder as log.txt. The menu at the
    top right ("...") has "Save diagnostics", which bundles the log and
    timings into one file to send for support.

To uninstall, use Settings -> Apps -> Installed apps -> FieldStitch.
Your projects in Documents\FieldStitch Projects are kept. A program log
is in %LOCALAPPDATA%\FieldStitch\logs if something goes wrong.
