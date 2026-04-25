import type { HistoryRecord } from "../../src/types/history";

interface Props {
  data: HistoryRecord[];
  currentSort: { by: string; order: "asc" | "desc" };
  onSort: (column: string) => void;
}

export const DetectionTable = ({ data, currentSort, onSort }: Props) => {
  const renderSortIcon = (column: string) => {
    if (currentSort.by !== column)
      return <span className="ml-1 opacity-20">↕</span>;
    return currentSort.order === "asc" ? (
      <span className="ml-1 text-blue-400">↑</span>
    ) : (
      <span className="ml-1 text-blue-400">↓</span>
    );
  };

  const columns = [
    { id: "id", label: "ID" },
    { id: "filename", label: "Plik" },
    { id: "model_version", label: "Model" },
    { id: "inference_time_ms", label: "Czas (ms)" },
    { id: "detected_count", label: "Znaków" },
    { id: "created_at", label: "Data" },
  ];

  return (
    <div className="w-full max-w-4xl mt-12 bg-gray-800 p-6 rounded-xl shadow-2xl border border-gray-700">
      <h2 className="text-xl font-bold mb-4 text-blue-400">Ewidencja Badań</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-gray-700/50 text-gray-300 uppercase text-xs">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.id}
                  onClick={() => onSort(col.id)}
                  className="px-4 py-4 cursor-pointer hover:bg-gray-600 transition-colors select-none group"
                >
                  <div className="flex items-center">
                    {col.label}
                    {renderSortIcon(col.id)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {data.map((record) => (
              <tr
                key={record.id}
                className="hover:bg-gray-700/30 transition-colors"
              >
                <td className="px-4 py-3 text-white font-medium">
                  {record.id}
                </td>
                <td className="px-4 py-3 truncate max-w-[150px]">
                  {record.filename}
                </td>
                <td className="px-4 py-3 text-blue-300 font-mono text-[11px]">
                  {record.model_version}
                </td>
                <td className="px-4 py-3 font-mono text-yellow-500">
                  {record.inference_time_ms} ms
                </td>
                <td className="px-4 py-3">{record.detected_count}</td>
                <td className="px-4 py-3 text-[11px] text-gray-500">
                  {new Date(record.created_at).toLocaleString("pl-PL")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
