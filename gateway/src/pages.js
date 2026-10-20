import VIEWER from "./viewer.gen.js"; // generated from src/hwb/assets/index.html by `npm run embed`

const SLOT = '<script id="hwb-report" type="application/json">__HWB_REPORT__</script>';

/** The hwb viewer with a stored report embedded, same as hwb's local report.html. */
export function reportPage(bundle, shared) {
  // Inside <script> only "</script" or "<!--" could end the element early; escaping every "<"
  // rules both out, and JSON.parse turns < back into "<".
  const data = JSON.stringify({ ...bundle, shared }).replace(/</g, "\\u003c");
  return VIEWER.replace(SLOT, SLOT.replace("__HWB_REPORT__", () => data));
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function plainPage(title, body) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)}</title>
<style>
body{margin:0;font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#111;background:#fff}
main{max-width:640px;margin:12vh auto;padding:0 16px}
h1{font-size:28px;margin:0 0 8px}p{margin:0 0 12px;color:#444}code{font:14px ui-monospace,Menlo,monospace}
</style></head><body><main>${body}</main></body></html>`;
}

export function landingPage() {
  return plainPage(
    "hwb gateway",
    `<h1>hwb gateway</h1>
<p>Shares <a href="https://github.com/autonomous-ai/hwb">hwb</a> GPU server acceptance reports.
Anyone can submit with <code>hwb submit</code>; host names, addresses, serials and paths are
removed first. Reports are unlisted: each one is reachable only through the link printed when it
was submitted, and expires after 90 days.</p>`,
  );
}

export function notFoundPage() {
  return plainPage(
    "Report not found",
    `<h1>Report not found</h1>
<p>This report does not exist, has been deleted by its uploader, or has expired.</p>`,
  );
}
