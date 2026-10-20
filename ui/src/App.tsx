import { useCallback, useEffect, useMemo, useState } from "react";
import { buildViews, type Bundle } from "./bundle";
import { Pill, ReportView, href } from "./components/ReportView";
import { Tabs } from "./components/Tabs";

interface Route {
  view?: string;
  tab?: string;
  check?: string;
}

// Route lives in the hash (#/<view>/<tab>/<check>) so it works from file:// and survives reloads.
function parseHash(hash: string): Route {
  const [view, tab, check] = hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  return { view: view || undefined, tab: tab || undefined, check: check || undefined };
}

export function App({ bundle }: { bundle: Bundle }) {
  const views = useMemo(() => buildViews(bundle), [bundle]);
  const [route, setRoute] = useState<Route>(() => parseHash(location.hash));

  useEffect(() => {
    const onHash = () => setRoute(parseHash(location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((view: string, tab: string, check?: string) => {
    const next = href(view, tab, check);
    if (location.hash !== next) history.pushState(null, "", next);
    setRoute({ view, tab, check });
  }, []);

  const view = views.find((v) => v.id === route.view) ?? views[0];
  useEffect(() => {
    if (view) document.title = `hwb · ${view.report.host} · ${view.label}`;
  }, [view]);
  if (!view) return <p className="load-error">This report contains no runs.</p>;

  return (
    <>
      <nav className="topnav">
        <span className="brand">hwb</span>
        <Tabs
          items={views.map((v) => ({
            id: v.id,
            label: (
              <>
                <span className={`sq ${v.status}`} aria-hidden="true" />
                {v.label}
              </>
            ),
          }))}
          active={view.id}
          onSelect={(id) => go(id, route.tab ?? "summary")}
          label="Runs"
          idPrefix="run"
          className="run-tabs"
        />
        <Pill status={view.status} />
      </nav>
      <div role="tabpanel" id={`run-panel-${view.id.replace(/[^\w-]/g, "_")}`} aria-labelledby={`run-tab-${view.id.replace(/[^\w-]/g, "_")}`}>
        <ReportView key={view.id} view={view} tab={route.tab ?? "summary"} check={route.check} go={go} shared={bundle.shared} />
      </div>
    </>
  );
}
