export const Badge = ({
  label,
  confidence,
}: {
  label: string;
  confidence: number;
}) => (
  <span
    style={{
      backgroundColor: "#3b82f6",
      color: "white",
      fontSize: "12px",
      fontWeight: "bold",
      position: "absolute",
      top: "-25px",
      left: "0",
      padding: "2px 6px",
      borderRadius: "4px 4px 0 0",
      whiteSpace: "nowrap",
    }}
  >
    {label} {Math.round(confidence * 100)}%
  </span>
);
