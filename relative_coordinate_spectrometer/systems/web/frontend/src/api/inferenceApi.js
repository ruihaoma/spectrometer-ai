const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8010/api";

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === "object" ? payload.detail || payload.message || "请求失败" : payload;
    throw new Error(message);
  }
  return payload;
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return parseResponse(response);
}

export async function predictImage({ file, sourceType, exposureStatus, note }) {
  const form = new FormData();
  form.append("file", file);
  form.append("source_type", sourceType);
  form.append("exposure_status", exposureStatus);
  form.append("note", note || "");

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: form,
  });
  return parseResponse(response);
}
