import { useState, useEffect } from "react";
import { Text } from "../components/atoms/Text";
import { Button } from "../components/atoms/Button";
import { ModelSelector } from "../components/molecules/ModelSelector";
import { ImageCanvas } from "../components/organisms/ImageCanvas";
import { DetectionTable } from "../components/organisms/DetectionTable";
import { predictImage } from "../api/detection";
import { getHistory } from "../api/history";
import { Pagination } from "../components/molecules/Pagination";
import type { Detection } from "./types/detection";

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("CNN_48_v1");
  const [history, setHistory] = useState([]);

  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [pageSize] = useState(10);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [sort, setSort] = useState({
    by: "created_at",
    order: "desc" as "asc" | "desc",
  });

  useEffect(() => {
    refreshHistory(1);
  }, []);
  useEffect(() => {
    refreshHistory();
  }, []);

  const refreshHistory = async (page: number = 1, sortParams = sort) => {
    try {
      const data = await getHistory(
        page,
        pageSize,
        sortParams.by,
        sortParams.order,
      );
      setHistory(data.items);
      setTotalPages(data.pages);
      setCurrentPage(data.page);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setDetections([]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    try {
      const result = await predictImage(selectedFile, model);
      setDetections(result.detections);
      await refreshHistory();
    } catch (err) {
      alert("Błąd serwera detekcji");
    } finally {
      setLoading(false);
    }
  };

  const getDetectedSignsSummary = () => {
    if (detections.length === 0) return "Nie wykryto żadnych znaków.";

    const labels = Array.from(
      new Set(detections.map((d: Detection) => d.label)),
    );

    return `Wykryto (${detections.length}): ${labels.join(", ")}`;
  };

  const handleSort = (column: string) => {
    const isAsc = sort.by === column && sort.order === "asc";

    const newOrder: "asc" | "desc" = isAsc ? "desc" : "asc";

    const newSort = { by: column, order: newOrder };

    setSort(newSort);
    refreshHistory(1, newSort);
  };
  return (
    <div className="min-h-screen bg-gray-900 text-white p-8 flex flex-col items-center">
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
          Klasyfikator Znaków Drogowych v1.1
        </h1>
        <p className="text-gray-500 mt-2">System wizyjny czasu rzeczywistego</p>
      </header>

      <main className="bg-gray-800 p-8 rounded-2xl shadow-2xl border border-gray-700 w-full max-w-4xl flex flex-col items-center">
        <ModelSelector value={model} onChange={setModel} />

        <div className="w-full mb-8">
          <input
            type="file"
            onChange={handleFileChange}
            className="block text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600/10 file:text-blue-400 hover:file:bg-blue-600/20 cursor-pointer"
            accept="image/*"
          />
        </div>

        {previewUrl && (
          <div className="flex flex-col items-center">
            <ImageCanvas imageUrl={previewUrl} detections={detections} />

            <div className="mt-4 bg-gray-900/50 px-4 py-2 rounded-lg border border-gray-700 w-full text-center">
              <Text variant={detections.length > 0 ? "success" : "secondary"}>
                {getDetectedSignsSummary()}
              </Text>
            </div>
          </div>
        )}

        <Button
          onClick={handleAnalyze}
          loading={loading}
          disabled={!selectedFile || loading}
        >
          Uruchom sieć neuronową
        </Button>
      </main>

      <section className="w-full max-w-4xl">
        <DetectionTable data={history} currentSort={sort} onSort={handleSort} />
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={(page) => refreshHistory(page)}
        />
      </section>
    </div>
  );
}

export default App;
