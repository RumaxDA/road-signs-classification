import { useRef, useState } from "react";
import { Badge } from "../atoms/Badge";
import type { Detection } from "../../src/types/detection";

export const ImageCanvas = ({
  imageUrl,
  detections,
}: {
  imageUrl: string;
  detections: Detection[];
}) => {
  const imageRef = useRef<HTMLImageElement>(null);
  const [_, setTick] = useState(0);

  return (
    <div
      style={{
        position: "relative",
        display: "inline-block",
        border: "2px solid #3b82f6",
        borderRadius: "8px",
        overflow: "hidden",
      }}
    >
      <img
        ref={imageRef}
        src={imageUrl}
        onLoad={() => setTick((t) => t + 1)}
        style={{ maxWidth: "100%", display: "block" }}
      />

      {imageRef.current?.complete &&
        detections.map((det, i) => {
          const img = imageRef.current!;
          const scaleX = img.clientWidth / img.naturalWidth;
          const scaleY = img.clientHeight / img.naturalHeight;

          return (
            <div
              key={i}
              style={{
                position: "absolute",
                border: "3px solid #3b82f6",
                left: `${det.box[0] * scaleX}px`,
                top: `${det.box[1] * scaleY}px`,
                width: `${(det.box[2] - det.box[0]) * scaleX}px`,
                height: `${(det.box[3] - det.box[1]) * scaleY}px`,
                pointerEvents: "none",
              }}
            >
              <Badge label={det.label} confidence={det.confidence} />
            </div>
          );
        })}
    </div>
  );
};
