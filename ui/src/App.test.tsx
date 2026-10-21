import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "./App";
import { buildViews, mergeRuns, normalize, type Bundle } from "./bundle";
import { shellQuote } from "./components/Sections";
import bench from "./fixtures/bench.json";
import broken from "./fixtures/broken.json";
import healthy from "./fixtures/healthy.json";
import install from "./fixtures/install.json";
import { humanSeconds, siBytes, verdict } from "./format";
import { entityColor } from "./components/Bench";
import { LineChart } from "./components/Chart";
import type { Report, Row } from "./types";

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

describe("benchmarks", () => {
  // Synthetic but realistic: 2 sockets + 2 GPUs, generated by hwb's own Timeseries code.
  const BENCH = bench as unknown as Report;

  it("shows headline numbers, charts per quantity and the phase table", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(BENCH)} />);
    await user.click(tab("Report sections", "Benchmarks"));
    const panel = activePanel();
    expect(within(panel).getByText("26,411")).toBeTruthy();
    expect(within(panel).getByText("77%")).toBeTruthy();
    const titles = Array.from(panel.querySelectorAll(".chart-title")).map((e) => e.textContent);
    expect(titles).toContain("CPU temperature");
    expect(titles).toContain("GPU temperature");
    expect(titles.filter((x) => x === "CPU temperature")).toHaveLength(2); // cpu + thermal benches
    // Never two units on one chart: CPU and GPU temperatures are separate charts.
    const cpuTemp = Array.from(panel.querySelectorAll("figure.chart")).find(
      (f) => f.querySelector(".chart-title")?.textContent === "CPU temperature",
    )!;
    expect(Array.from(cpuTemp.querySelectorAll(".legend li")).map((l) => l.textContent)).toEqual([
      "Package id 0",
      "Package id 1",
    ]);
    expect(cpuTemp.querySelector(".limit text")?.textContent).toBe("limit 100 °C");
    expect(cpuTemp.querySelectorAll("path.line")).toHaveLength(2);
    const phases = Array.from(panel.querySelectorAll("table.phase-table tbody tr"));
    expect(phases.map((r) => r.querySelector("td")?.textContent)).toEqual([
      "idle",
      "single-core",
      "all-core",
      "cooldown",
      "idle",
      "load",
      "cooldown",
    ]);
  });

  it("reads values at the crosshair with the keyboard", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(BENCH)} />);
    await user.click(tab("Report sections", "Benchmarks"));
    const svg = activePanel().querySelector<SVGSVGElement>("figure.chart svg")!;
    svg.focus();
    await user.keyboard("{ArrowRight}");
    const tip = activePanel().querySelector(".tooltip")!;
    expect(tip.querySelector(".tt-time")?.textContent).toBe("0:00 · idle");
    expect(tip.querySelectorAll(".tt-row")).toHaveLength(2);
    expect(tip.textContent).toMatch(/Package id 0\d+\.\d °C/);
  });

  it("colours follow the entity on every chart", () => {
    const sensors = ["cpu.temp.Package id 0", "cpu.temp.Package id 1"];
    expect(entityColor("gpu1.temp", sensors)).toBe(entityColor("gpu1.power", sensors));
    expect(entityColor("gpu0.util", sensors)).not.toBe(entityColor("gpu1.util", sensors));
    expect(entityColor("cpu.temp.Package id 1", sensors)).not.toBe(entityColor("cpu.temp.Package id 0", sensors));
    expect(entityColor("mem.used", sensors)).not.toBe(entityColor("cpu.util", sensors));
  });

  it("overview keeps benchmarks from separate runs", () => {
    const cpuOnly = { ...BENCH, started_at: "2026-10-20T08:00:00+00:00", sections: { bench: { cpu: (BENCH.sections.bench as Row).cpu } } };
    const thermalOnly = {
      ...BENCH,
      started_at: "2026-10-20T09:00:00+00:00",
      sections: { bench: { thermal: (BENCH.sections.bench as Row).thermal } },
    };
    const { report } = mergeRuns(
      [
        { id: "a", report: cpuOnly },
        { id: "b", report: thermalOnly },
      ],
      BENCH.host,
    );
    expect(Object.keys(report.sections.bench as Row).sort()).toEqual(["cpu", "thermal"]);
  });
});

describe("chart scales", () => {
  it("never zooms into sensor noise", async () => {
    const { yScale } = await import("./components/Chart");
    const s = yScale([95.0, 95.2, 95.6], "°C");
    expect(s.hi - s.lo).toBeGreaterThanOrEqual(20);
    expect(s.lo).toBeLessThanOrEqual(95);
    expect(s.hi).toBeGreaterThanOrEqual(95.6);
    expect(yScale([0, 3], "%")).toMatchObject({ lo: 0, hi: 100 });
    const w = yScale([2.0, 3.0], "W");
    expect(w.lo).toBe(0); // small non-negative values start at zero
  });
});

describe("PCIe link monitoring", () => {
  const t = [0, 1, 2, 3, 4, 5];
  const monitor: Report = {
    ...HEALTHY,
    command: "monitor",
    results: [{ name: "bench.monitor", status: "warn", summary: "GPU1 PCIe link at Gen3 (max Gen4) while busy", hint: "reseat", data: {} }],
    sections: {
      bench: {
        monitor: {
          seconds: 6,
          cpu_util_pct: 41,
          findings: [{ level: "warn", text: "GPU1 PCIe link at Gen3 (max Gen4) while busy, in 3 of 6 busy samples", hint: "reseat" }],
          pcie: {
            "1": { addr: "0000:41:00.0", upstream: "0000:40:01.1", max_gen: 4, max_width: 16, min_width: 8, correctable_errors: 12, uncorrectable_errors: 0, replays: null },
          },
          timeseries: {
            t,
            interval_s: 1,
            phases: [{ name: "monitor", start: 0, end: 5 }],
            series: [
              { key: "gpu1.util", label: "GPU1 utilization", group: "utilization", unit: "%", values: [99, 99, 99, 99, 99, 99] },
              { key: "gpu1.pcie_gen", label: "GPU1 PCIe generation", group: "pcie", unit: "gen", values: [4, 4, 4, 3, 3, 3] },
              { key: "gpu1.pcie_errors", label: "GPU1 correctable PCIe errors", group: "errors", unit: "errors", values: [0, 0, 0, 5, 9, 12] },
            ],
          },
        },
      },
    },
  };

  it("charts the link generation as steps and lists each GPU link", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(monitor)} />);
    await user.click(tab("Report sections", "Benchmarks"));
    const panel = activePanel();
    expect(within(panel).getByRole("heading", { name: "Monitor" })).toBeTruthy();
    const gen = Array.from(panel.querySelectorAll("figure.chart")).find(
      (f) => f.querySelector(".chart-title")?.textContent === "GPU PCIe link generation",
    )!;
    expect(Array.from(gen.querySelectorAll(".tick")).map((x) => x.textContent)).toContain("Gen4");
    expect(gen.querySelector("path.line")!.getAttribute("d")).toContain("H"); // stepped
    const row = panel.querySelector("table.pcie-table tbody tr")!;
    expect(row.textContent).toContain("0000:41:00.0");
    expect(row.querySelector("td.bad")?.textContent).toBe("x8");
    expect(within(panel).getByText(/added no load/)).toBeTruthy();
  });
});

describe("model serving", () => {
  const serveReport: Report = {
    ...HEALTHY,
    command: "serve",
    results: [
      { name: "serve.vllm", status: "pass", summary: "ready", hint: null, data: {} },
      { name: "serve.checks", status: "fail", summary: "1 of 5 failed", hint: "parser", data: {} },
      { name: "serve.load", status: "pass", summary: "2100 tok/s", hint: null, data: {} },
    ],
    sections: {
      bench: {
        serve: {
          mode: "launch",
          model: "deepseek-v4-flash-vision",
          ready_s: 412,
          preset: { name: "deepseek-v4-flash-vision", status: "verified", model: "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp" },
          command: ["vllm", "serve", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "--tensor-parallel-size", "2"],
          checks: [
            { name: "arithmetic", status: "pass", answer: "391", ttft_s: 0.21 },
            { name: "tool_call", status: "fail", detail: "no tool call (check --tool-call-parser)", answer: "It is sunny.", ttft_s: 0.3 },
          ],
          load: [
            { concurrency: 1, requests: 4, ok: 4, errors: 0, output_tok_per_s: 180.4, per_request_tok_per_s: 182, ttft_ms_p50: 210, ttft_ms_p99: 260, tpot_ms_p50: 5.5, max_stall_ms: 40 },
            { concurrency: 16, requests: 64, ok: 64, errors: 0, output_tok_per_s: 2100.2, per_request_tok_per_s: 140, ttft_ms_p50: 900, ttft_ms_p99: 1800, tpot_ms_p50: 7.1, max_stall_ms: 120 },
          ],
        },
      },
    },
  };

  it("shows the model, best throughput, checks and the load table", async () => {
    const user = userEvent.setup();
    render(<App bundle={normalize(serveReport)} />);
    await user.click(tab("Report sections", "Benchmarks"));
    const panel = activePanel();
    expect(within(panel).getByRole("heading", { name: "Model serving" })).toBeTruthy();
    expect(within(panel).getByText("2,100 tok/s")).toBeTruthy();
    expect(within(panel).getByText("1 / 2")).toBeTruthy(); // checks passed / counted
    expect(within(panel).getByText(/no tool call/)).toBeTruthy();
    const rows = panel.querySelectorAll("table.serve-load tbody tr");
    expect(rows).toHaveLength(2);
    expect(rows[1]!.textContent).toContain("2,100");
    expect(within(panel).getByText(/vllm serve deepseek-ai/)).toBeTruthy();
  });
});

describe("models on the GPUs", () => {
  it("lists running model servers in the inventory", async () => {
    const user = userEvent.setup();
    const inv = (HEALTHY.sections.inventory ?? {}) as Row;
    const report: Report = {
      ...HEALTHY,
      sections: {
        ...HEALTHY.sections,
        inventory: {
          ...inv,
          workloads: {
            servers: [{ framework: "vllm", model: "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", served_name: "dsv4", port: 8000, gpus: ["0", "1"], used_gib: 180, processes: 2, serving: { ids: ["dsv4"] } }],
            processes: [{ gpu: "0", pid: 1240, process: "python3.12", used_mib: 92160 }],
          },
        },
      },
    };
    render(<App bundle={normalize(report)} />);
    await user.click(tab("Report sections", "Inventory"));
    const card = Array.from(activePanel().querySelectorAll("section.card")).find((c) => c.querySelector("h3")?.textContent === "Models on the GPUs")!;
    expect(card.textContent).toContain("deepseek-ai/DeepSeek-V4-Flash-Vision-Exp");
    expect(card.textContent).toContain("0, 1");
    expect(card.textContent).toContain("vllm :8000 → dsv4");
  });
});

describe("model weights on disk", () => {
  it("lists cached models in the inventory", async () => {
    const user = userEvent.setup();
    const inv = (HEALTHY.sections.inventory ?? {}) as Row;
    const report: Report = {
      ...HEALTHY,
      sections: {
        ...HEALTHY.sections,
        inventory: {
          ...inv,
          model_weights: {
            models: [{ id: "Qwen/Qwen3.8-27B", source: "huggingface", size_gb: 55.6, architecture: "Qwen3_5ForConditionalGeneration", format: "safetensors", complete: true }],
          },
        },
      },
    };
    render(<App bundle={normalize(report)} />);
    await user.click(tab("Report sections", "Inventory"));
    const card = Array.from(activePanel().querySelectorAll("section.card")).find((c) => c.querySelector("h3")?.textContent?.startsWith("Model weights on disk"))!;
    expect(card.textContent).toContain("Qwen/Qwen3.8-27B");
    expect(card.textContent).toContain("Qwen3_5ForConditionalGeneration");
  });
});

// jsdom has no PointerEvent; without it pointerType never reaches the handler.
if (typeof window.PointerEvent === "undefined") {
  class PointerEventShim extends MouseEvent {
    pointerType: string;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerType = init.pointerType ?? "mouse";
    }
  }
  (window as unknown as { PointerEvent: typeof PointerEventShim }).PointerEvent = PointerEventShim;
}

describe("charts on phones", () => {
  const props = { title: "Clock", unit: "MHz", t: [100, 110, 120, 130, 140], lines: [{ key: "a", label: "CPU", color: "#000", values: [1, 2, 3, 4, 5] }], phases: [] };

  it("labels the time axis as elapsed time from 0:00", () => {
    const { container } = render(<LineChart {...props} />);
    const ticks = Array.from(container.querySelectorAll("text.tick")).map((e) => e.textContent);
    expect(ticks).toContain("0:00");
    expect(ticks).not.toContain("1:40"); // absolute seconds (t0 = 100) must not leak into the labels
  });

  it("keeps a tapped value shown when the finger lifts", () => {
    const { container } = render(<LineChart {...props} />);
    const svg = container.querySelector("svg")!;
    fireEvent.pointerDown(svg, { clientX: 100, pointerType: "touch" });
    expect(container.querySelector(".tooltip")).toBeTruthy();
    fireEvent.pointerLeave(svg, { pointerType: "touch" });
    expect(container.querySelector(".tooltip")).toBeTruthy();
    fireEvent.pointerLeave(svg, { pointerType: "mouse" });
    expect(container.querySelector(".tooltip")).toBeNull();
  });
});
