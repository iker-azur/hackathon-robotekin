/**
 * Why a task counts as done.
 *
 * The live path used to decide "done" from two readings -- some force was present, and the
 * part ended up in the tray. Both can be true of a grasp that never happened: a gripper
 * pressed on a corner reports plenty of force, and a part punted across the table lands in
 * the tray. Those two checks together let a pick be reported as successful that no robot
 * actually performed.
 *
 * So a completed task now carries the full record: seven criteria, each a reading, each
 * with its number. Three of them -- `pads_on_faces`, `part_rose`, `carried` -- are what
 * makes it mean what it says, and they are the ones that were missing.
 *
 * This panel exists so that claim is checkable rather than asserted. A judge can read the
 * 29.40 mm span against the part's 30 mm width, the 60.9 N against the 0.31 N the friction
 * cone needs, and the 98.9 mm rise, and see that the grip is real.
 */

import React from "react";
import type { Evidence } from "./api";
import { t, num, type Lang } from "./i18n";

const CRITERIA_LABELS: Record<string, string> = {
  tracking: "evTracking",
  pads_on_faces: "evPads",
  grip: "evGrip",
  part_rose: "evRose",
  carried: "evCarried",
  part_in_bin: "evInBin",
  part_resting: "evResting",
};

const CRITERION_ORDER = [
  "tracking", "pads_on_faces", "grip", "part_rose", "carried", "part_in_bin", "part_resting",
];

/** Exported so the metrics panel labels the same seven criteria identically. */
export const CRITERION_LABEL_KEYS: Record<string, string> = {
  tracking: "evTracking",
  pads_on_faces: "evPads",
  grip: "evGrip",
  part_rose: "evRose",
  carried: "evCarried",
  part_in_bin: "evInBin",
  part_resting: "evResting",
};

type Props = { lang: Lang; evidence: Evidence | null; note: string };

export default function EvidencePanel({ lang, evidence, note }: Props) {
  if (!evidence) {
    return (
      <div className="evidence empty-ev">
        <p>{t(lang, "noEvidence")}</p>
        {note && <p className="note">{note}</p>}
      </div>
    );
  }
  const m = evidence.measurements;

  return (
    <div className="evidence">
      <div className="ev-verdict" data-ok={evidence.ok}>
        <strong>{evidence.ok ? "✓" : "✗"}</strong>
        <span>{evidence.ok
          ? (lang === "es" ? "agarre demostrado" : "grasp demonstrated")
          : (lang === "es" ? "agarre no demostrado" : "grasp not demonstrated")}</span>
        {evidence.robotId && <span className="ev-by">{evidence.robotId}</span>}
      </div>

      <ul className="criteria">
        {CRITERION_ORDER.map((key) => {
          const ok = evidence.criteria[key];
          return (
            <li key={key} className={ok ? "ok" : "bad"}>
              <i>{ok ? "✓" : "✗"}</i>
              <span>{t(lang, CRITERIA_LABELS[key])}</span>
              <em>
                {key === "tracking" && `${num(lang, m.toolErrorM * 1000, 2)} mm`}
                {key === "pads_on_faces" &&
                  `${num(lang, m.spanOnPartM * 1000, 2)} / ${num(lang, m.partWidthM * 1000, 1)} mm`}
                {key === "grip" &&
                  `${num(lang, m.gripForceN, 1)} N` +
                  (m.gripMargin != null ? ` · ×${num(lang, m.gripMargin, 0)}` : "")}
                {key === "part_rose" && `${num(lang, m.partRiseM * 1000, 1)} mm`}
                {key === "carried" && (m.partCarriedToBin ? "✓" : "—")}
                {key === "part_in_bin" && (m.partInBin ? "✓" : "—")}
                {key === "part_resting" && (m.partResting ? "✓" : "—")}
              </em>
            </li>
          );
        })}
      </ul>

      {m.gripMargin != null && (
        <p className="ev-foot">
          {t(lang, "gripMargin")}: ×{num(lang, m.gripMargin, 0)} &nbsp;
          <span className="dim">
            {num(lang, m.gripForceN, 1)} N measured ·{" "}
            {num(lang, m.gripRequiredN, 2)} N required by the friction cone
          </span>
        </p>
      )}

      {evidence.failures.length > 0 && (
        <ul className="ev-failures">
          {evidence.failures.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      )}
    </div>
  );
}