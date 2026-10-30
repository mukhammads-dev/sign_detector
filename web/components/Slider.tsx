"use client";

type Props = {
  name: string;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  note?: string;
  onChange: (value: number) => void;
};

export default function Slider({
  name,
  value,
  display,
  min,
  max,
  step,
  note,
  onChange,
}: Props) {
  return (
    <div className="control">
      <div className="row">
        <span className="name">{name}</span>
        <span className="value">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {note ? <div className="note">{note}</div> : null}
    </div>
  );
}
