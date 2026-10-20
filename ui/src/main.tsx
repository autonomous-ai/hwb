import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { normalize, type Bundle } from "./bundle";
import "./styles.css";
import type { Report } from "./types";

const PLACEHOLDER = "__HWB_REPORT__";

async function load(): Promise<Bundle | null> {
  const text = document.getElementById("hwb-report")?.textContent?.trim() ?? "";
  if (text && text !== PLACEHOLDER) return normalize(JSON.parse(text));
  if (import.meta.env.DEV) {
    // `npm run dev`: preview every fixture as one output directory.
    const fixtures = import.meta.glob<{ default: Report }>("./fixtures/*.json", { eager: true });
    return {
      runs: Object.entries(fixtures).map(([path, mod]) => ({
        id: `${mod.default.started_at}-${path.replace(/^.*\/|\.json$/g, "")}`,
        report: mod.default,
      })),
    };
  }
  return null;
}

const root = createRoot(document.getElementById("root")!);
load().then(
  (bundle) =>
    root.render(
      <StrictMode>
        {bundle ? <App bundle={bundle} /> : <p className="load-error">No report data is embedded in this page.</p>}
      </StrictMode>,
    ),
  (err: unknown) => root.render(<p className="load-error">Could not read the report data: {String(err)}</p>),
);
