"use client";

import { useRef, useState } from "react";

type Props = {
  preview: string | null;
  onFile: (file: File) => void;
};

export default function Dropzone({ preview, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  function take(files: FileList | null) {
    const file = files?.[0];
    if (file && file.type.startsWith("image/")) onFile(file);
  }

  return (
    <div
      className={`dropzone${over ? " over" : ""}`}
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        take(e.dataTransfer.files);
      }}
    >
      <input
        ref={input}
        type="file"
        accept="image/*"
        onChange={(e) => take(e.target.files)}
      />
      {preview ? (
        // The preview is what the model actually saw: resized, and with the
        // stickers already applied. Showing the original would hide the attack.
        <img src={preview} alt="what the model sees" />
      ) : (
        <div className="hint">
          <strong>Drop a traffic sign image</strong>
          or click to choose a file
        </div>
      )}
    </div>
  );
}
