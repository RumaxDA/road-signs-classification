export interface HistoryRecord {
  id: number;
  filename: string;
  model_version: string;
  inference_time_ms: number;
  detected_count: number;
  created_at: string;
}
