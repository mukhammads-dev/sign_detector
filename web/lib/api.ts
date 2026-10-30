export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type TopK = { label: string; prob: number };

export type Prediction = {
  accepted: boolean;
  label: string;
  confidence: number;
  threshold: number;
  temperature: number;
  energy: number;
  occlusion_requested: number;
  occlusion_actual: number;
  preview: string;
  topk: TopK[];
};

export type Meta =
  | { ready: false; error: string | null }
  | {
      ready: true;
      classes: string[];
      temperature: number;
      threshold: number;
      test_acc: number | null;
      selective_acc: number | null;
      coverage: number | null;
    };

export async function fetchMeta(): Promise<Meta> {
  const res = await fetch(`${API_URL}/meta`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API returned ${res.status}`);
  return res.json();
}

export async function predict(
  file: File,
  threshold: number,
  occlusion: number,
  seed: number
): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);
  body.append("threshold", String(threshold));
  body.append("occlusion", String(occlusion));
  body.append("seed", String(seed));

  const res = await fetch(`${API_URL}/predict`, { method: "POST", body });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `API returned ${res.status}`);
  }
  return res.json();
}
