import { useRef, type KeyboardEvent, type ReactNode } from "react";

export interface TabItem {
  id: string;
  label: ReactNode;
}

/** WAI-ARIA tabs: arrow keys / Home / End move between tabs and select them. */
export function Tabs({
  items,
  active,
  onSelect,
  label,
  idPrefix,
  className,
}: {
  items: TabItem[];
  active: string;
  onSelect: (id: string) => void;
  label: string;
  idPrefix: string;
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, i: number) => {
    const last = items.length - 1;
    const next =
      e.key === "ArrowRight" ? (i === last ? 0 : i + 1)
      : e.key === "ArrowLeft" ? (i === 0 ? last : i - 1)
      : e.key === "Home" ? 0
      : e.key === "End" ? last
      : null;
    if (next === null) return;
    e.preventDefault();
    refs.current[next]?.focus();
    onSelect(items[next]!.id);
  };
  return (
    <div role="tablist" aria-label={label} className={className}>
      {items.map((t, i) => {
        const selected = t.id === active;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={tabId(idPrefix, t.id)}
            aria-selected={selected}
            aria-controls={panelId(idPrefix, t.id)}
            tabIndex={selected ? 0 : -1}
            onClick={() => onSelect(t.id)}
            onKeyDown={(e) => onKey(e, i)}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

export function TabPanel({
  idPrefix,
  id,
  active,
  children,
}: {
  idPrefix: string;
  id: string;
  active: boolean;
  children: ReactNode;
}) {
  // Inactive panels stay in the DOM (hidden) so printing can show every section.
  return (
    <section role="tabpanel" id={panelId(idPrefix, id)} aria-labelledby={tabId(idPrefix, id)} hidden={!active} className="panel">
      {children}
    </section>
  );
}

const safe = (s: string) => s.replace(/[^\w-]/g, "_");
const tabId = (p: string, id: string) => `${p}-tab-${safe(id)}`;
const panelId = (p: string, id: string) => `${p}-panel-${safe(id)}`;
