import { Cpu } from "lucide-react";

export default function ModelInfoPanel({ health }) {
  return (
    <section className="tool-panel">
      <h2 className="panel-title">
        <Cpu size={15} />
        模型设置区
      </h2>
      <dl className="grid gap-2 text-[12px]">
        <div>
          <dt className="text-lab-muted">模型名称</dt>
          <dd className="field-value mt-1">spectrum_unet_transformer_1d</dd>
        </div>
        <div>
          <dt className="text-lab-muted">模型文件路径</dt>
          <dd className="field-value mt-1 break-all">{health?.model_path || "未加载最后一次 8万 checkpoint"}</dd>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <dt className="text-lab-muted">输入 shape</dt>
            <dd className="field-value mt-1">[1, 4, 2501]</dd>
          </div>
          <div>
            <dt className="text-lab-muted">输出 shape</dt>
            <dd className="field-value mt-1">[1, 2501]</dd>
          </div>
        </div>
        <div>
          <dt className="text-lab-muted">推理设备</dt>
          <dd className="field-value mt-1">{health?.device || "CPU / CUDA 自动"}</dd>
        </div>
      </dl>
    </section>
  );
}
