"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import Dropzone from "@/components/Dropzone";
import Slider from "@/components/Slider";
import Verdict from "@/components/Verdict";
import { fetchMeta, predict, type Meta, type Prediction } from "@/lib/api";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [threshold, setThreshold] = useState(0.9);
  const [occlusion, setOcclusion] = useState(0);
  const [seed, setSeed] = useState(0);

  // the request currently in flight, so a slow response cannot overwrite a
  // newer one when the slider is dragged quickly
  const latest = useRef(0);

  useEffect(() => {
    fetchMeta()
      .then((m) => {
        setMeta(m);
        if (m.ready) setThreshold(m.threshold);
      })
      .catch((e: Error) => setMetaError(e.message));
  }, []);

  const run = useCallback(
    async (f: File, occ: number, s: number) => {
      const ticket = ++latest.current;
      setBusy(true);
      setError(null);
      try {
        const r = await predict(f, threshold, occ, s);
        if (ticket === latest.current) setResult(r);
      } catch (e) {
        if (ticket === latest.current) setError((e as Error).message);
      } finally {
        if (ticket === latest.current) setBusy(false);
      }
    },
    [threshold]
  );

  function onFile(f: File) {
    setFile(f);
    setOcclusion(0);
    run(f, 0, seed);
  }

  // Occlusion changes the image, so the model has to run again. Debounced so
  // dragging the slider does not fire a request per pixel.
  useEffect(() => {
    if (!file) return;
    const t = setTimeout(() => run(file, occlusion, seed), 220);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [occlusion, seed]);

  // The threshold does not change the model output, only the decision, so it
  // is applied locally and the verdict flips with no round trip.
  const view: Prediction | null = result
    ? { ...result, threshold, accepted: result.confidence >= threshold }
    : null;

  const notReady = meta && !meta.ready;

  return (
    <main className="page">
      <header className="header">
        <h1>SignGuard AI</h1>
        <p>
          A traffic sign classifier with a gate in front of it. Below the
          confidence threshold it answers <strong>UNKNOWN</strong> instead of
          guessing. Drag the sliders to see what the gate costs and what it
          buys.
        </p>
      </header>

      {metaError && (
        <div className="banner">
          Cannot reach the API at <code>{process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}</code>.
          Start it with <code>cd api &amp;&amp; uvicorn main:app --port 8000</code>.
        </div>
      )}

      {notReady && (
        <div className="banner">
          The API is running but has no model yet. Train one with a notebook
          from <code>notebooks/</code> and put{" "}
          <code>signguard_calibrated.pth</code> in <code>models/</code>, then
          restart the API.
        </div>
      )}

      <div className="layout">
        <div>
          <div className="card">
            <h2>Input — what the model sees</h2>
            <Dropzone preview={view?.preview ?? null} onFile={onFile} />
          </div>

          {error && <div className="banner">{error}</div>}

          {view && (
            <div className="card">
              <h2>Decision</h2>
              <Verdict result={view} />
            </div>
          )}
        </div>

        <aside>
          <div className="card">
            <h2>Controls</h2>

            <Slider
              name="Confidence gate"
              value={threshold}
              display={pct(threshold)}
              min={0}
              max={0.999}
              step={0.001}
              note={
                meta?.ready
                  ? `Audited value: ${pct(meta.threshold)}. Raising it refuses more inputs and is right more often on the ones it keeps.`
                  : "Raising it refuses more inputs and is right more often on the ones it keeps."
              }
              onChange={setThreshold}
            />

            <Slider
              name="Sticker coverage"
              value={occlusion}
              display={
                view ? pct(view.occlusion_actual) : `${Math.round(occlusion * 100)}%`
              }
              min={0}
              max={0.5}
              step={0.01}
              note="Covers the sign with opaque patches, the way tape or a sticker would. Watch the confidence stay high while the label goes wrong — that is the failure the gate exists to catch."
              onChange={setOcclusion}
            />

            {occlusion > 0 && (
              <button
                className="control"
                onClick={() => setSeed((s) => s + 1)}
                style={{
                  marginTop: 14,
                  background: "#1f2630",
                  border: "1px solid #2d3542",
                  borderRadius: 9,
                  color: "#e6edf3",
                  padding: "8px 14px",
                  fontSize: 13,
                  cursor: "pointer",
                  width: "100%",
                }}
              >
                New sticker placement
              </button>
            )}
          </div>

          {busy && <div className="card spinner">running the model…</div>}

          {view && (
            <div className="card">
              <h2>Class probabilities</h2>
              <div className="bars">
                {view.topk.map((t) => (
                  <div className="bar" key={t.label}>
                    <span className="name" title={t.label}>
                      {t.label}
                    </span>
                    <span className="value">{(t.prob * 100).toFixed(1)}%</span>
                    <div className="track">
                      <div
                        className="fill"
                        style={{ width: `${Math.max(t.prob * 100, 0.6)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {view && (
            <div className="card">
              <h2>Diagnostics</h2>
              <dl className="facts">
                <div className="fact">
                  <dt>Calibrated confidence</dt>
                  <dd>{pct(view.confidence)}</dd>
                </div>
                <div className="fact">
                  <dt>Temperature</dt>
                  <dd>{view.temperature.toFixed(2)}</dd>
                </div>
                <div className="fact">
                  <dt>Energy score</dt>
                  <dd>{view.energy.toFixed(2)}</dd>
                </div>
                <div className="fact">
                  <dt>Area covered</dt>
                  <dd>{pct(view.occlusion_actual)}</dd>
                </div>
              </dl>
            </div>
          )}

          {meta?.ready && meta.test_acc != null && (
            <div className="card">
              <h2>Audited model</h2>
              <dl className="facts">
                <div className="fact">
                  <dt>Test accuracy, no gate</dt>
                  <dd>{pct(meta.test_acc)}</dd>
                </div>
                {meta.coverage != null && (
                  <div className="fact">
                    <dt>Answered, with gate</dt>
                    <dd>{pct(meta.coverage)}</dd>
                  </div>
                )}
                {meta.selective_acc != null && (
                  <div className="fact">
                    <dt>Accuracy on answers</dt>
                    <dd>{pct(meta.selective_acc)}</dd>
                  </div>
                )}
              </dl>
            </div>
          )}
        </aside>
      </div>

      <p className="footnote">
        The temperature and the threshold come from the audit notebook, not from
        this page. The sticker attack runs the same code the audit uses, so what
        you see here and the numbers in the README are produced identically.
      </p>
    </main>
  );
}
