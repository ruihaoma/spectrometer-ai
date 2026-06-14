function triggerDownload(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function downloadJson(filename, data) {
  triggerDownload(filename, JSON.stringify(data, null, 2), "application/json;charset=utf-8");
}

export function downloadCsv(filename, rows) {
  const csv = rows
    .map((row) => row.map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(","))
    .join("\n");
  triggerDownload(filename, csv, "text/csv;charset=utf-8");
}

export function downloadText(filename, content) {
  triggerDownload(filename, content, "text/plain;charset=utf-8");
}
