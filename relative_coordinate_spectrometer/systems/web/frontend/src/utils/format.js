export function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

export function fileFormat(filename) {
  const suffix = filename.split(".").pop();
  return suffix ? suffix.toLowerCase() : "未知";
}

export function buildPredictionRows(result) {
  if (!result) return [];
  const rows = [["wavelength_nm", "R", "G", "B", "Gray", "spectrum_pred"]];
  const wavelength = result.wavelength_nm || [];
  const profile = result.profile || {};
  const spectrum = result.prediction?.spectrum_pred || [];
  wavelength.forEach((nm, index) => {
    rows.push([
      nm,
      profile.R?.[index] ?? "",
      profile.G?.[index] ?? "",
      profile.B?.[index] ?? "",
      profile.Gray?.[index] ?? "",
      spectrum[index] ?? "",
    ]);
  });
  return rows;
}

export function buildReport(result, fileInfo) {
  if (!result) return "";
  return [
    "# 智能光谱重建预测报告",
    "",
    `- 文件名：${fileInfo?.name || result.original_image_info?.filename || ""}`,
    `- 光源类型：${result.experiment_info?.source_type || ""}`,
    `- 曝光状态：${result.experiment_info?.exposure_status || ""}`,
    `- 模型名称：${result.model_info?.model_name || ""}`,
    `- 模型路径：${result.model_info?.model_path || ""}`,
    `- 定标版本：${result.processing_info?.calibration_version || ""}`,
    `- 主峰波长：${formatNumber(result.prediction?.peak_wavelength_nm, 3)} nm`,
    `- 峰值强度：${formatNumber(result.prediction?.peak_intensity, 6)}`,
    "",
    "## 风险提示",
    "",
    ...(result.warnings || []),
    "",
    "## 处理日志",
    "",
    ...(result.logs || []),
  ].join("\n");
}
