interface Props {
  value: string;
  onChange: (val: string) => void;
}

export const ModelSelector = ({ value, onChange }: Props) => (
  <div className="mb-6 w-full flex items-center justify-between bg-gray-700/50 p-4 rounded-lg border border-gray-600">
    <label className="font-semibold text-gray-300 text-sm">
      Parametry wnioskowania:
    </label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-gray-900 text-white px-3 py-2 rounded border border-gray-600 focus:border-blue-500 outline-none text-sm cursor-pointer transition-colors"
    >
      <option value="YOLO+CNN_48_v1">Hybryda (YOLO + CNN 48px)</option>
      <option value="YOLO_Only">Szybki (Tylko YOLO)</option>
      <option value="CNN_224_v3">Precyzyjny (CNN 224px)</option>
    </select>
  </div>
);
