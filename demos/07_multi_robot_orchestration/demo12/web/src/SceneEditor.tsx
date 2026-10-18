/**
 * Editing the cell.
 *
 * The honesty problem here is the whole design constraint. Isaac Lab cannot insert an
 * articulation into a live scene, and the in-process rebuild was **measured to hang** at
 * 370 CPU-seconds without returning -- `RemovePrim` plus a second `sim.reset()` in the same
 * Kit session. So the number of arms, the parts and the trays are fixed when the process
 * starts, and this panel cannot pretend otherwise.
 *
 * What it does instead:
 *
 *  * validates on every edit, so an impossible layout is refused while you are still
 *    holding it rather than three minutes into a simulator boot;
 *  * says plainly that applying restarts the process, and why;
 *  * keeps the *task plan* live and instant, because a task is orchestration state and
 *    changes nothing physical.
 *
 * The task queue survives the restart through the ledger. Part positions do not: parts
 * respawn, so a task marked done is re-checked against the physics and requeued if the
 * claim no longer holds.
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Scene } from "./api";
import { t, num, type Lang } from "./i18n";

type Props = {
  lang: Lang;
  scene: Scene | null;
  onApplied: () => void;
};

export default function SceneEditor({ lang, scene, onApplied }: Props) {
  const [draft, setDraft] = useState<Scene | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [checking, setChecking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (scene && !draft) setDraft(structuredClone(scene));
  }, [scene, draft]);

  // Re-validate whenever the draft changes, debounced so typing does not flood the API.
  useEffect(() => {
    if (!draft) return;
    setChecking(true);
    const handle = window.setTimeout(() => {
      api.validateScene(draft)
        .then((r) => setErrors(r.errors ?? []))
        .catch((e) => setErrors([String(e)]))
        .finally(() => setChecking(false));
    }, 250);
    return () => window.clearTimeout(handle);
  }, [draft]);

  const setRobots = useCallback((n: number) => {
    setDraft((d) => {
      if (!d) return d;
      return { ...d, robots: d.robots.map((r, i) => ({ ...r, enabled: i < n })) };
    });
  }, []);

  const apply = useCallback(async () => {
    if (!draft) return;
    setBusy(true);
    try {
      const res = await api.applyScene(draft);
      if (res.ok) {
        setNote({ ok: true, text: lang === "es" ? "Reiniciando…" : "Restarting…" });
      } else {
        setNote({
          ok: false,
          text: res.reason ?? (res.errors ?? []).join("; "),
        });
      }
    } catch (e) {
      setNote({ ok: false, text: String(e) });
    } finally {
      setBusy(false);
    }
  }, [draft, lang]);

  if (!draft) {
    return <p className="hint">{lang === "es" ? "Cargando escena…" : "Loading scene…"}</p>;
  }

  const enabled = draft.robots.filter((r) => r.enabled).length;
  const valid = errors.length === 0;

  return (
    <div className="editor">
      <p className="hint">{t(lang, "editorHint")}</p>

      <div className="row">
        <label className="lbl">{t(lang, "robots")}:</label>
        {[1, 2, 3].map((n) => (
          <button key={n} className={enabled === n ? "primary" : ""}
                  onClick={() => setRobots(n)}>
            {n}
          </button>
        ))}
        <span className="dim">
          {draft.parts.length} {t(lang, "pieces")} · {draft.bins.length} {t(lang, "trays_filled")}
        </span>
      </div>

      <div className="row">
        <span className="lbl">{t(lang, "pieces")}:</span>
        <span className="chips">
          {draft.parts.map((p) => (
            <span key={p.id} className="chip" title={`${p.massG} g`}>
              <i style={{ background: p.color }} />
              {p.id.replace("part_", "P")} {num(lang, p.sizeM[0] * 1000, 0)}
            </span>
          ))}
        </span>
      </div>

      <div className="row">
        <span className="lbl">{t(lang, "trays")}:</span>
        <span className="chips">
          {draft.bins.map((b) => (
            <span key={b.id} className="chip">
              <i style={{ background: b.color }} />{b.label}
            </span>
          ))}
        </span>
      </div>

      <div className={`validation ${valid ? "ok" : "bad"}`}>
        <strong>
          {checking
            ? (lang === "es" ? "Comprobando…" : "Checking…")
            : valid ? `✓ ${t(lang, "valid")}` : `✗ ${t(lang, "invalid")}`}
        </strong>
        <ul>
          {errors.map((e, i) => <li key={i}>{e}</li>)}
        </ul>
      </div>

      <div className="row">
        <button className="primary" disabled={busy || !valid}
                onClick={() => void apply()}>
          {t(lang, "applyScene")}
        </button>
        <button disabled={busy} onClick={() => setDraft(structuredClone(scene as Scene))}>
          {lang === "es" ? "Descartar cambios" : "Discard changes"}
        </button>
      </div>

      {note && <p className={`msg ${note.ok ? "ok" : "bad"}`}>{note.text}</p>}
    </div>
  );
}