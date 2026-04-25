export interface Detection {
  box: [number, number, number, number];
  label: string;
  confidence: number;
}
