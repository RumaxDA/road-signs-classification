export const predictImage = async (file: File, model: string) => {
  const formData = new FormData();
  formData.append("file", file);
  const url = new URL("http://localhost:8000/detection/predict");
  url.searchParams.append("model_version", model);

  const response = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });
  return response.json();
};
