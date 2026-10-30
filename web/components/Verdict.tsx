"use client";

import type { Prediction } from "@/lib/api";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function Verdict({ result }: { result: Prediction }) {
  const { accepted, label, confidence, threshold } = result;

  return (
    <div className={`verdict ${accepted ? "accepted" : "rejected"}`}>
      <div className="state">{accepted ? "answered" : "refused"}</div>
      <div className="label">{accepted ? label : "UNKNOWN"}</div>
      <div className="sub">
        {accepted ? (
          <>
            {pct(confidence)} calibrated confidence, above the {pct(threshold)}{" "}
            gate.
          </>
        ) : (
          <>
            Closest match <strong>{label}</strong> at {pct(confidence)}, below
            the {pct(threshold)} gate — so nothing is returned. In a car this
            frame would go to another sensor or to the driver.
          </>
        )}
      </div>
    </div>
  );
}
