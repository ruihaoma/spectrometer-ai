import { Play, RotateCcw } from "lucide-react";

export default function ActionPanel({ status, canPredict, onPredict, onClearResult }) {
  const isProcessing = status === "处理中";
  const actionLabel = isProcessing ? "处理中" : status === "预测成功" ? "重新预测" : "开始预测";

  return (
    <section className="tool-panel bg-[#f8fbff]">
      <h2 className="panel-title">
        <Play size={15} />
        推理控制区
      </h2>
      <div className="grid grid-cols-1 gap-2">
        <button className="primary-button h-10 text-[14px]" type="button" onClick={onPredict} disabled={!canPredict || isProcessing}>
          <Play size={15} />
          {actionLabel}
        </button>
        <button className="control-button" type="button" onClick={onClearResult}>
          <RotateCcw size={15} />
          清空结果
        </button>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-[12px]">
        {["待上传", "处理中", "预测成功", "预测失败"].map((item) => (
          <div
            key={item}
            className={`rounded border px-2 py-1.5 text-center ${
              item === status ? "border-lab-blue bg-lab-blueSoft font-semibold text-lab-blue" : "border-lab-line bg-white text-lab-muted"
            }`}
          >
            {item}
          </div>
        ))}
      </div>
    </section>
  );
}
