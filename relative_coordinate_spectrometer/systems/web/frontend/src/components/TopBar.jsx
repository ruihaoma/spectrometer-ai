import { Activity, Cpu, FileText, Play, Upload } from "lucide-react";
import { formatNumber } from "../utils/format.js";

const statusStyle = {
  待上传: "border-lab-line bg-white text-lab-muted",
  处理中: "border-[#b98500] bg-[#fff7df] text-[#7a5200]",
  预测成功: "border-[#88c7a2] bg-[#ecfdf3] text-lab-success",
  预测失败: "border-[#f2b8b5] bg-[#fff0ef] text-lab-danger",
};

const acceptedImageTypes = ".png,.jpg,.jpeg,.bmp,.tif,.tiff,image/png,image/jpeg,image/bmp,image/tiff";

export default function TopBar({ status, health, fileInfo, result, error, canPredict, hasResult, onFileChange, onPredict, onReport }) {
  const isProcessing = status === "处理中";
  const actionLabel = isProcessing ? "处理中" : status === "预测成功" ? "重新预测" : "开始预测";
  const backendReady = Boolean(health?.model_loaded);
  const peakLabel = result ? `${formatNumber(result.prediction?.peak_wavelength_nm, 3)} nm` : "--";

  return (
    <header className="sticky top-0 z-20 flex min-h-16 flex-col gap-3 border-b border-lab-line bg-white px-5 py-3 shadow-[0_1px_0_rgba(23,32,51,0.04)] md:flex-row md:items-center md:justify-between">
      <div className="min-w-0">
        <h1 className="text-[18px] font-semibold leading-6 text-lab-ink md:text-[21px] md:leading-7">
          智能光谱重建预测系统 v1.0
        </h1>
        <p className="text-[12px] leading-5 text-lab-muted">
          上传 DIY 拍摄光谱图像，系统自动提取四通道光谱输入，并通过神经网络预测标准光谱曲线。
        </p>
      </div>
      <div className="flex w-full flex-wrap items-center gap-2 md:w-auto md:justify-end">
        <div
          className={`hidden h-8 max-w-[240px] items-center gap-2 rounded border px-3 text-[12px] xl:flex ${
            fileInfo ? "border-lab-line bg-lab-field text-lab-ink" : "border-lab-line bg-white text-lab-muted"
          }`}
          title={fileInfo?.name || "未选择文件"}
        >
          <span className="h-2 w-2 rounded-full bg-lab-blue" />
          <span className="truncate">{fileInfo?.name || "未选择文件"}</span>
        </div>
        <div className="hidden items-center gap-2 text-[12px] text-lab-muted md:flex">
          <Cpu size={15} />
          <span>{backendReady ? health?.device || "cpu" : "未加载"}</span>
        </div>
        <div
          className={`hidden h-8 min-w-[120px] items-center justify-center rounded border px-3 text-[12px] font-semibold lg:flex ${
            error
              ? "border-[#f2b8b5] bg-[#fff0ef] text-lab-danger"
              : result
                ? "border-[#88c7a2] bg-[#ecfdf3] text-lab-success"
                : "border-lab-line bg-white text-lab-muted"
          }`}
          title={error || (result ? "预测主峰波长" : "等待预测结果")}
        >
          {error ? "有错误" : result ? `主峰 ${peakLabel}` : "等待结果"}
        </div>
        <label className="inline-flex h-10 cursor-pointer items-center justify-center gap-2 rounded border border-lab-line bg-white px-4 text-[13px] font-semibold text-lab-ink transition hover:bg-lab-blueSoft">
          <Upload size={15} />
          <span>选择照片</span>
          <input
            className="hidden"
            type="file"
            accept={acceptedImageTypes}
            onChange={(event) => {
              onFileChange(event.target.files?.[0] || null);
              event.target.value = "";
            }}
          />
        </label>
        <button
          className="inline-flex h-10 min-w-[124px] items-center justify-center gap-2 rounded border border-lab-blue bg-lab-blue px-4 text-[14px] font-semibold text-white shadow-sm transition hover:bg-[#174b86] disabled:cursor-not-allowed disabled:border-lab-line disabled:bg-[#d7dee8] disabled:text-lab-muted"
          type="button"
          onClick={onPredict}
          disabled={!canPredict || isProcessing}
          aria-label={actionLabel}
          title={canPredict ? actionLabel : "请先选择 DIY 光谱照片"}
        >
          <Play size={15} />
          <span>{actionLabel}</span>
        </button>
        <button
          className="inline-flex h-10 items-center justify-center gap-2 rounded border border-lab-line bg-white px-4 text-[13px] font-semibold text-lab-ink transition hover:bg-lab-blueSoft disabled:cursor-not-allowed disabled:opacity-50"
          type="button"
          onClick={onReport}
          disabled={!hasResult}
          aria-label="导出报告"
          title={hasResult ? "导出报告" : "预测完成后可导出报告"}
        >
          <FileText size={15} />
          <span>报告</span>
        </button>
        <div
          className={`flex h-8 min-w-[94px] items-center justify-center gap-2 whitespace-nowrap rounded border px-3 text-[13px] font-semibold ${statusStyle[status]}`}
        >
          <Activity size={15} />
          <span>{status}</span>
        </div>
      </div>
    </header>
  );
}
