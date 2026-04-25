interface Props {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export const Pagination = ({
  currentPage,
  totalPages,
  onPageChange,
}: Props) => {
  if (totalPages <= 1) return null;

  return (
    <div className="flex items-center space-x-2 mt-4 justify-center">
      <button
        disabled={currentPage === 1}
        onClick={() => onPageChange(currentPage - 1)}
        className="px-3 py-1 bg-gray-700 rounded disabled:opacity-30 hover:bg-gray-600 transition"
      >
        Poprzednia
      </button>

      <span className="text-sm text-gray-400">
        Strona <span className="text-white font-bold">{currentPage}</span> z{" "}
        {totalPages}
      </span>

      <button
        disabled={currentPage === totalPages}
        onClick={() => onPageChange(currentPage + 1)}
        className="px-3 py-1 bg-gray-700 rounded disabled:opacity-30 hover:bg-gray-600 transition"
      >
        Następna
      </button>
    </div>
  );
};
