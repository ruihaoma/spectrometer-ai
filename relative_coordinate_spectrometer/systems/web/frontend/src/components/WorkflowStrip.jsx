import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";

function stepClass(state) {
  if (state === "error") return "border-[#f2b8b5] bg-[#fff0ef] text-lab-danger";
  if (state === "done") return "border-[#88c7a2] bg-[#ecfdf3] text-lab-success";
  if (state === "active") return "border-[#f0c66c] bg-[#fff8e6] text-[#7a5200]";
  return "border-lab-line bg-white text-lab-muted";
}

function StepIcon({ state }) {
  if (state === "error") return <XCircle size={16} />;
  if (state === "done") return <CheckCircle2 size={16} />;
  if (state === "active") return <Loader2 className="animate-spin" size={16} />;
  return <Circle size={16} />;
}

export default function WorkflowStrip({ status, fileInfo, result, error }) {
  const isProcessing = status === "处理中";
  const steps = [
    {
      label: "图片",
      detail: fileInfo ? fileInfo.name : "待上传",
      state: fileInfo ? "done" : "todo",
    },
    {
      label: "ROI",
      detail: result?.roi_image_base64 ? "已裁剪" : isProcessing ? "提取中" : "等待",
      state: error ? "error" : result?.roi_image_base64 ? "done" : isProcessing ? "active" : "todo",
    },
    {
      label: "四通道",
      detail: result?.profile ? "已生成" : isProcessing ? "生成中" : "等待",
      state: error ? "error" : result?.profile ? "done" : isProcessing ? "active" : "todo",
    },
    {
      label: "预测",
      detail: result?.prediction ? "已完成" : isProcessing ? "推理中" : "等待",
      state: error ? "error" : result?.prediction ? "done" : isProcessing ? "active" : "todo",
    },
  ];

  return (
    <section className="mb-4 grid grid-cols-2 gap-2 lg:grid-cols-4">
      {steps.map((step) => (
        <div key={step.label} className={`flex min-h-14 items-center gap-3 rounded-md border px-3 py-2 ${stepClass(step.state)}`}>
          <StepIcon state={step.state} />
          <div className="min-w-0">
            <div className="text-[13px] font-semibold leading-5">{step.label}</div>
            <div className="truncate text-[12px] leading-4 opacity-85" title={step.detail}>
              {step.detail}
            </div>
          </div>
        </div>
      ))}
    </section>
  );
}
