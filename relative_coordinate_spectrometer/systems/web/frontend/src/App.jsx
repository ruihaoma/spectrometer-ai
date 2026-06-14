import { useEffect, useMemo, useState } from "react";
import { fetchHealth, predictImage } from "./api/inferenceApi.js";
import ControlPanel from "./components/ControlPanel.jsx";
import ImageCard from "./components/ImageCard.jsx";
import LogPanel from "./components/LogPanel.jsx";
import PredictionChart from "./components/PredictionChart.jsx";
import ProfileChart from "./components/ProfileChart.jsx";
import ResultSummary from "./components/ResultSummary.jsx";
import RiskNotice from "./components/RiskNotice.jsx";
import TopBar from "./components/TopBar.jsx";
import WorkflowStrip from "./components/WorkflowStrip.jsx";
import { downloadCsv, downloadJson, downloadText } from "./utils/download.js";
import { buildPredictionRows, buildReport } from "./utils/format.js";

const supportedImagePattern = /\.(png|jpe?g|bmp|tiff?)$/i;

function readImageSize(file, objectUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("无法读取本地图片尺寸"));
    image.src = objectUrl;
  });
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState("待上传");
  const [file, setFile] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [sourceType, setSourceType] = useState("未知");
  const [exposureStatus, setExposureStatus] = useState("自动");
  const [note, setNote] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err) => {
        setHealth({ warnings: [`后端健康检查失败：${err.message}`], model_loaded: false, device: "unknown" });
      });
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const logs = useMemo(() => {
    if (result?.logs?.length) return result.logs;
    if (error) return [`预测失败：${error}`];
    return [];
  }, [error, result]);

  function isSupportedImage(file) {
    return Boolean(file?.type?.startsWith("image/") || supportedImagePattern.test(file?.name || ""));
  }

  async function handleFileChange(nextFile) {
    if (!nextFile) return;
    if (!isSupportedImage(nextFile)) {
      setError("请选择 png、jpg、bmp、tif 或 tiff 格式的光谱图片");
      setStatus("预测失败");
      return;
    }
    const objectUrl = URL.createObjectURL(nextFile);
    try {
      const size = await readImageSize(nextFile, objectUrl);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setFile(nextFile);
      setPreviewUrl(objectUrl);
      setFileInfo({ name: nextFile.name, size: nextFile.size, width: size.width, height: size.height });
      setStatus("待上传");
      setError("");
      setResult(null);
    } catch (err) {
      URL.revokeObjectURL(objectUrl);
      setError(err.message);
      setStatus("预测失败");
    }
  }

  function eventHasFiles(event) {
    return Array.from(event.dataTransfer?.types || []).includes("Files");
  }

  function handleDragEnter(event) {
    if (!eventHasFiles(event)) return;
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragOver(event) {
    if (!eventHasFiles(event)) return;
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave(event) {
    if (event.clientX <= 0 || event.clientY <= 0 || event.currentTarget === event.target) {
      setDragActive(false);
    }
  }

  function handleDrop(event) {
    if (!eventHasFiles(event)) return;
    event.preventDefault();
    setDragActive(false);
    const nextFile = Array.from(event.dataTransfer?.files || []).find(isSupportedImage);
    handleFileChange(nextFile || null);
    if (!nextFile) {
      setError("拖入的文件不是支持的图片格式");
      setStatus("预测失败");
    }
  }

  function clearFile() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setFileInfo(null);
    setPreviewUrl("");
    setResult(null);
    setError("");
    setStatus("待上传");
  }

  function clearResult() {
    setResult(null);
    setError("");
    setStatus("待上传");
  }

  async function handlePredict() {
    if (!file) return;
    setStatus("处理中");
    setError("");
    setResult(null);
    try {
      const payload = await predictImage({ file, sourceType, exposureStatus, note });
      setResult(payload);
      setStatus("预测成功");
      fetchHealth().then(setHealth).catch(() => {});
    } catch (err) {
      setError(err.message);
      setStatus("预测失败");
      fetchHealth().then(setHealth).catch(() => {});
    }
  }

  function exportCsv() {
    downloadCsv("prediction.csv", buildPredictionRows(result));
  }

  function exportJson() {
    const payload = result
      ? {
          wavelength_nm: result.wavelength_nm,
          profile: result.profile,
          spectrum_pred: result.prediction?.spectrum_pred,
          peak_wavelength_nm: result.prediction?.peak_wavelength_nm,
          peak_intensity: result.prediction?.peak_intensity,
          model_info: result.model_info,
          processing_info: result.processing_info,
          experiment_info: result.experiment_info,
          warnings: result.warnings,
          logs: result.logs,
        }
      : {};
    downloadJson("prediction.json", payload);
  }

  function exportReport() {
    downloadText("prediction_report.md", buildReport(result, fileInfo));
  }

  return (
    <div
      className="min-h-screen bg-[#eef2f7] text-lab-ink lg:flex lg:h-screen lg:flex-col lg:overflow-hidden"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {dragActive ? (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-[#172033]/45 p-6">
          <div className="rounded-md border border-white/70 bg-white px-8 py-6 text-center shadow-xl">
            <div className="text-[18px] font-semibold text-lab-ink">松开上传光谱图片</div>
            <div className="mt-2 text-[13px] text-lab-muted">支持 png、jpg、bmp、tif、tiff</div>
          </div>
        </div>
      ) : null}
      <TopBar
        status={status}
        health={health}
        fileInfo={fileInfo}
        result={result}
        error={error}
        canPredict={Boolean(file)}
        hasResult={Boolean(result)}
        onFileChange={handleFileChange}
        onPredict={handlePredict}
        onReport={exportReport}
      />
      <main className="flex flex-col lg:min-h-0 lg:flex-1 lg:flex-row">
        <ControlPanel
          health={health}
          status={status}
          fileInfo={fileInfo}
          sourceType={sourceType}
          exposureStatus={exposureStatus}
          note={note}
          canPredict={Boolean(file)}
          hasResult={Boolean(result)}
          onFileChange={handleFileChange}
          onClearFile={clearFile}
          onSourceType={setSourceType}
          onExposureStatus={setExposureStatus}
          onNote={setNote}
          onPredict={handlePredict}
          onClearResult={clearResult}
          onCsv={exportCsv}
          onJson={exportJson}
          onReport={exportReport}
        />
        <section className="flex min-w-0 flex-1 flex-col overflow-auto p-4">
          <WorkflowStrip status={status} fileInfo={fileInfo} result={result} error={error} />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ImageCard
              title="原始 DIY 图像"
              imageUrl={previewUrl}
              placeholder="请先选择 DIY 光谱照片"
              stateLabel={fileInfo ? "已选择" : "待上传"}
            />
            <ImageCard
              title="ROI 裁剪结果"
              imageUrl={result?.roi_image_base64}
              placeholder="等待 ROI 提取"
              loading={status === "处理中"}
              stateLabel={result?.roi_image_base64 ? "已提取" : status === "处理中" ? "处理中" : "等待"}
            />
            <ProfileChart result={result} status={status} />
            <PredictionChart result={result} status={status} />
          </div>
          <div className="mt-4 grid gap-4">
            <ResultSummary result={result} status={status} sourceType={sourceType} exposureStatus={exposureStatus} />
            <RiskNotice warnings={result?.warnings} error={error} health={health} />
            <LogPanel logs={logs} />
          </div>
        </section>
      </main>
    </div>
  );
}
