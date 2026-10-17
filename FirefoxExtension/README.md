# Firefox media links

WaifuFinder can receive direct media links from Firefox. The add-on only checks
top-level HTTP(S) links and forwards configured suffixes such as `.mp4`, `.dms`,
`.ts`, and `.m3u8`. It does not inspect embedded players or send browsing data to
a service.

## Set up

1. In WaifuFinder, open **Settings → Firefox** and edit the suffix list if needed.
2. Click **Set Up / Update Firefox**. WaifuFinder registers its local bridge,
   opens Firefox's add-on loader, and highlights the packaged add-on file.
3. In Firefox, click **Load Temporary Add-on…** and select the highlighted
   `WaifuFinderFirefox.xpi` file.
4. Return to WaifuFinder. The status turns green when Firefox connects.

The add-on's toolbar popup shows connection status. Registration, suffixes, and
unregistration are managed in WaifuFinder; there is no separate add-on Register
button.

Firefox removes temporary add-ons when it exits, so this local setup must be
repeated after a restart. Permanent installation in standard Firefox requires a
Mozilla-signed package. Unregister in WaifuFinder to stop forwarding and remove
the local bridge. Remove the temporary add-on itself from Firefox's Add-ons
Manager if desired.

The bridge accepts messages only from the fixed add-on ID, validates HTTP(S)
URLs and suffixes again, and opens WaifuFinder through its registered URL scheme.
