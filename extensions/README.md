# Browser Extensions

This directory contains browser extensions for Chrome, Firefox, and Edge that integrate with the Extension API service to provide real-time phishing detection.

## Browser Support

- **Chrome** (primary target): Full feature parity, including navigation blocking and `declarativeNetRequest`-based proactive blocking.
- **Edge**: Full parity with Chrome. Edge is Chromium-based and supports the same APIs.
- **Firefox**: Full blocking support via `webNavigation`; no `declarativeNetRequest` (MV2). Limited support compared to Chrome—use Chrome for the best experience.

## Structure

- `chrome/` - Chrome extension (Manifest V3)
- `firefox/` - Firefox extension (Manifest V2)
- `edge/` - Edge extension (Manifest V3)

## Installation

### Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `chrome/` directory
5. Configure API URL and API key in extension options

### Firefox

1. Open Firefox and navigate to `about:debugging`
2. Click "This Firefox"
3. Click "Load Temporary Add-on"
4. Select `firefox/manifest.json`
5. Configure API URL and API key in extension options

### Edge

1. Open Edge and navigate to `edge://extensions/`
2. Enable "Developer mode" (toggle in bottom left)
3. Click "Load unpacked"
4. Select the `edge/` directory
5. Configure API URL and API key in extension options

## Configuration

1. Right-click the extension icon and select "Options"
2. Enter your Extension API URL (default: `http://localhost:3003/api/v1/extension`)
3. Enter your API key
4. Optionally enable "Block critical threats automatically"
5. Click "Save Settings"

## Features

- **Real-time URL checking**: Automatically checks URLs when pages load
- **Navigation blocking**: Redirects to `blocked.html` when threats meet severity threshold (Chrome/Edge: `declarativeNetRequest`; Firefox: `webNavigation`)
- **Proceed anyway / allowlist**: Users can temporarily bypass a block; URL is removed from blocklist
- **Form detection**: Detects suspicious forms with password fields
- **Visual warnings**: Shows warning banners for detected threats
- **Threat reporting**: Allows users to report suspected phishing sites
- **Local caching**: Caches results for 30 minutes to reduce API calls
- **Badge indicators**: Shows warning badge on extension icon for threats

## API Requirements

The extension requires:
- Extension API service running (default: `http://localhost:3003`)
- Valid API key for authentication
- CORS configured to allow extension origins

## Icons

Replace the placeholder icon files in each extension's `icons/` directory with actual icons:
- `icon16.png` - 16x16 pixels
- `icon48.png` - 48x48 pixels
- `icon128.png` - 128x128 pixels

## Development

To modify the extensions:

1. Make changes to the JavaScript files
2. Reload the extension in your browser
3. Test the changes
4. For Chrome/Edge: Use the "Reload" button in `chrome://extensions/` or `edge://extensions/`
5. For Firefox: Use "Reload" in `about:debugging`

## Notes

- The extensions use the same codebase with minor API differences (Chrome/Edge use `chrome.*`, Firefox uses `browser.*`)
- Manifest versions differ: Chrome/Edge use V3, Firefox uses V2
- Service workers are used for Chrome/Edge, background scripts for Firefox
- Edge and Firefox load API URL/API key from storage correctly; defaults to `http://localhost:3003/api/v1/extension` when unset
