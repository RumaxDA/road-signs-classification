export const getHistory = async (
  page: number,
  size: number,
  sortBy: string,
  order: string,
) => {
  const url = new URL("http://localhost:8000/detection/history");
  url.searchParams.append("page", page.toString());
  url.searchParams.append("size", size.toString());
  url.searchParams.append("sort_by", sortBy);
  url.searchParams.append("order", order);

  const response = await fetch(url.toString());
  if (!response.ok) throw new Error("Błąd pobierania");
  return response.json();
};
