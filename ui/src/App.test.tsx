import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "./App";
import { buildViews, mergeRuns, normalize, type Bundle } from "./bundle";
import { shellQuote } from "./components/Sections";
import broken from "./fixtures/broken.json";
import healthy from "./fixtures/healthy.json";
import install from "./fixtures/install.json";
import { humanSeconds, siBytes, verdict } from "./format";
import type { Report } from "./types";

// Fixtures are real report.json files from tests/test_flow.py's simulated rigs, run in the
// order all (healthy, warnings) -> detect (broken driver) -> install (dry run).
const HEALTHY = healthy as Report;
const BROKEN = broken as Report;
const INSTALL = install as Report;
const ALL_RUNS: Bundle = {
  runs: [
    { id: "run-all", report: HEALTHY },
    { id: "run-detect", report: BROKEN },
    { id: "run-install", report: INSTALL },
  ],
};

beforeEach(() => history.replaceState(null, "", "/"));

const activePanel = () => document.querySelector<HTMLElement>(".panel:not([hidden])")!;
const checkNames = () =>
  Array.from(activePanel().querySelectorAll("article.check")).map((el) => el.querySelector("code")!.textContent);
const tab = (list: string, name: RegExp | string) =>
  within(screen.getByRole("tablist", { name: list })).getByRole("tab", { name });
const selected = (list: string, name: RegExp | string) => tab(list, name).getAttribute("aria-selected") === "true";

describe("single run", () => {
  it("shows verdict, attention and per-section tabs", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(HEALTHY)} />);
    expect(screen.queryByRole("tab", { name: /Overview/ })).toBeNull();
    expect(screen.getByRole("status").textContent).toContain("Acceptable with warnings");
    const rows = within(activePanel().querySelector<HTMLElement>("table.attention")!).getAllByRole("row").slice(1);
    expect(rows.map((r) => r.querySelector("code")?.textContent)).toEqual([
      "gpu.kernel_headers",
      "gpu.pcie_link",
      "gpu.persistence",
      "gpu.xid",
    ]);

    await user.click(tab("Report sections", "Inventory"));
    expect(selected("Report sections", "Inventory")).toBe(true);
    expect(within(activePanel()).getByText("Gen5 x8/x16").classList.contains("bad")).toBe(true);
    expect(within(activePanel()).getByText("3.84 TB")).toBeTruthy();
    expect(within(activePanel()).getByText("71 °C").classList.contains("warm")).toBe(true);
    expect(location.hash).toBe("#/./inventory");
  });

  it("moves between section tabs with the keyboard", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(HEALTHY)} />);
    tab("Report sections", "Summary").focus();
    await user.keyboard("{ArrowRight}");
    expect(selected("Report sections", /Checks/)).toBe(true);
    expect(document.activeElement).toBe(tab("Report sections", /Checks/));
    await user.keyboard("{End}");
    expect(selected("Report sections", "Raw JSON")).toBe(true);
  });

  it("filters checks by status and by text", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(HEALTHY)} />);
    await user.click(tab("Report sections", /Checks/));
    expect(checkNames()).toHaveLength(HEALTHY.results.length);

    await user.click(within(activePanel()).getByRole("button", { name: /Pass/ }));
    expect(checkNames()).not.toContain("gpu.pci");
    expect(checkNames()).toContain("gpu.xid");

    await user.click(within(activePanel()).getByRole("button", { name: /Pass/ }));
    await user.type(within(activePanel()).getByRole("searchbox"), "persistence");
    expect(checkNames()).toEqual(["gpu.persistence"]);

    await user.clear(within(activePanel()).getByRole("searchbox"));
    await user.type(within(activePanel()).getByRole("searchbox"), "no such check");
    expect(within(activePanel()).getByText("No checks match the filter.")).toBeTruthy();
  });

  it("jumps from a summary count to the checks with that status", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(HEALTHY)} />);
    await user.click(within(activePanel()).getByRole("button", { name: /Warn/ }));
    expect(selected("Report sections", /Checks/)).toBe(true);
    expect(checkNames()).toEqual(["gpu.kernel_headers", "gpu.pcie_link", "gpu.persistence", "gpu.xid"]);
  });

  it("opens the checks tab at a check linked from the attention table", () => {
    render(<App bundle={normalize(HEALTHY)} />);
    const link = within(activePanel()).getByRole("link", { name: "gpu.xid" });
    history.replaceState(null, "", link.getAttribute("href")!);
    fireEvent(window, new HashChangeEvent("hashchange"));
    expect(selected("Report sections", /Checks/)).toBe(true);
    expect(activePanel().querySelector("article.check.focus code")?.textContent).toBe("gpu.xid");
  });

  it("expands and collapses all details", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(HEALTHY)} />);
    await user.click(tab("Report sections", /Checks/));
    const details = () => Array.from(activePanel().querySelectorAll<HTMLDetailsElement>("details.data"));
    expect(details().length).toBeGreaterThan(0);
    expect(details().every((d) => !d.open)).toBe(true);
    await user.click(within(activePanel()).getByRole("button", { name: "Expand all" }));
    expect(details().every((d) => d.open)).toBe(true);
    await user.click(within(activePanel()).getByRole("button", { name: "Collapse all" }));
    expect(details().every((d) => !d.open)).toBe(true);
  });

  it("renders the install plan as copyable commands", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(INSTALL)} />);
    await user.click(tab("Report sections", "Install plan"));
    expect(within(activePanel()).getByText("Install plan: driver")).toBeTruthy();
    expect(within(activePanel()).getByText(/nvidia-driver-580-server-open/, { selector: "pre.cmd" })).toBeTruthy();
    expect(within(activePanel()).getByText(/Dry run/)).toBeTruthy();
  });

  it("gives sections it has no dedicated view for their own tab", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize({ ...HEALTHY, sections: { bench_gpu: { gemm_tflops: { bf16: 209.5 } } } })} />);
    await user.click(tab("Report sections", "bench gpu"));
    expect(within(activePanel()).getByText("209.5")).toBeTruthy();
  });

  it("never turns machine strings into markup", () => {
    const evil = "<img src=x onerror=alert(1)>";
    const report: Report = {
      ...HEALTHY,
      host: evil,
      results: [{ name: "gpu.x", status: "warn", summary: evil, hint: evil, data: { name: evil } }],
    };
    const { container } = render(<App bundle={normalize(report)} />);
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getAllByText(evil).length).toBeGreaterThan(0);
  });
});

describe("several runs in one file", () => {
  it("has an overview tab plus one tab per run, newest first", () => {
    render(<App bundle={ALL_RUNS} />);
    const names = within(screen.getByRole("tablist", { name: "Runs" }))
      .getAllByRole("tab")
      .map((t) => t.textContent);
    expect(names).toEqual([
      "Overview",
      expect.stringMatching(/^install · /),
      expect.stringMatching(/^detect · /),
      expect.stringMatching(/^all · /),
    ]);
    expect(selected("Runs", "Overview")).toBe(true);
  });

  it("overview keeps the latest result of each check and says where it came from", async () => {
    const user = userEvent.setup();
    render(<App bundle={ALL_RUNS} />);
    // detect ran after all, so its driver failures replace all's passes.
    expect(screen.getByRole("status").textContent).toContain("Not acceptable");
    expect(within(activePanel()).getByText("Runs in this report")).toBeTruthy();
    await user.click(tab("Report sections", /Checks/));
    const nouveau = activePanel().querySelector<HTMLElement>("#check-gpu\\.nouveau")!;
    expect(nouveau.dataset.status).toBe("fail");
    expect(nouveau.querySelector(".source")?.textContent).toMatch(/^from detect · /);
    const cpu = activePanel().querySelector<HTMLElement>("#check-info\\.cpu")!;
    expect(cpu.querySelector(".source")?.textContent).toMatch(/^from all · /);
  });

  it("switches runs and keeps the section tab when the run has it", async () => {
    const user = userEvent.setup();
    render(<App bundle={ALL_RUNS} />);
    await user.click(tab("Report sections", "Inventory"));
    await user.click(tab("Runs", /^all · /));
    expect(screen.getByRole("status").textContent).toContain("Acceptable with warnings");
    expect(selected("Report sections", "Inventory")).toBe(true);
    await user.click(tab("Runs", /^install · /));
    expect(selected("Report sections", "Summary")).toBe(true);
  });

  it("merges per host", () => {
    const other = { ...HEALTHY, host: "gpu-rig-02" };
    const views = buildViews({ runs: [...ALL_RUNS.runs, { id: "run-other", report: other }] });
    expect(
      views
        .filter((v) => v.kind === "overview")
        .map((v) => v.label)
        .sort(),
    ).toEqual(["Overview · gpu-rig-01", "Overview · gpu-rig-02"]);
    expect(mergeRuns(ALL_RUNS.runs, "gpu-rig-02").report.results).toEqual([]);
  });
});

describe("format", () => {
  it("formats units", () => {
    expect(siBytes(3840755982336)).toBe("3.84 TB");
    expect(siBytes(512)).toBe("512 B");
    expect(humanSeconds(5.25)).toBe("5.3 s");
    expect(humanSeconds(93784)).toBe("1d 2h 3m");
    expect(shellQuote("a b")).toBe("'a b'");
    expect(shellQuote("DEBIAN_FRONTEND=noninteractive")).toBe("DEBIAN_FRONTEND=noninteractive");
  });

  it("derives the verdict from the worst status", () => {
    const c = { fail: 0, error: 0, warn: 0, pass: 1, skip: 0 };
    expect(verdict(c, 0).title).toBe("Acceptable");
    expect(verdict({ ...c, warn: 1 }, 0).title).toBe("Acceptable with warnings");
    expect(verdict({ ...c, error: 1 }, 2).title).toBe("Incomplete");
    expect(verdict({ ...c, fail: 1, error: 1 }, 1).title).toBe("Not acceptable");
    expect(verdict({ ...c, pass: 0, skip: 3 }, 0).title).toBe("Nothing verified");
  });
});

describe("shared on the web", () => {
  it("drops local file links and says what was removed", () => {
    render(
      <App
        bundle={{
          runs: [{ id: "run-all", report: HEALTHY }],
          shared: { id: "3f0c1d1e-0000-4000-8000-000000000000", submitted_at: "2026-10-20T07:00:00Z", label: "rack 7" },
        }}
      />,
    );
    const footer = document.querySelector("footer")!;
    expect(footer.textContent).toContain("Shared report 3f0c1d1e-0000-4000-8000-000000000000 (rack 7)");
    expect(footer.textContent).toContain("were removed before upload");
    expect(footer.querySelector('a[href$="report.json"]')).toBeNull();
  });
});
