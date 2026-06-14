import { SlidersHorizontal } from "lucide-react";

export default function ProcessingConfigPanel({ health }) {
  const calibration = health?.calibration_version || "relative_spectral_coordinate_linear_diagnostic";
  return (
    <section className="tool-panel">
      <h2 className="panel-title">
        <SlidersHorizontal size={15} />
        图像处理参数区
      </h2>
      <dl className="grid gap-2 text-[12px]">
        <div className="grid grid-cols-[92px_1fr] gap-2">
          <dt className="text-lab-muted">ROI 模式</dt>
          <dd className="font-medium text-lab-ink">相对彩色谱线自动 ROI</dd>
        </div>
        <div className="grid grid-cols-[92px_1fr] gap-2">
          <dt className="text-lab-muted">定标模型</dt>
          <dd className="break-all font-medium text-lab-ink">{calibration}</dd>
        </div>
        <div className="grid grid-cols-[92px_1fr] gap-2">
          <dt className="text-lab-muted">波长范围</dt>
          <dd className="font-medium text-lab-ink">400-650 nm</dd>
        </div>
        <div className="grid grid-cols-[92px_1fr] gap-2">
          <dt className="text-lab-muted">波长步长</dt>
          <dd className="font-medium text-lab-ink">0.1 nm</dd>
        </div>
        <div className="grid grid-cols-[92px_1fr] gap-2">
          <dt className="text-lab-muted">点数</dt>
          <dd className="font-medium text-lab-ink">2501</dd>
        </div>
        <div className="grid grid-cols-[92px_1fr] gap-2">
          <dt className="text-lab-muted">输入通道</dt>
          <dd className="font-medium text-lab-ink">R/G/B/Gray</dd>
        </div>
        <div className="grid grid-cols-[92px_1fr] gap-2">
          <dt className="text-lab-muted">归一化</dt>
          <dd className="font-medium text-lab-ink">每样本每通道 x/max(x)</dd>
        </div>
      </dl>
    </section>
  );
}
