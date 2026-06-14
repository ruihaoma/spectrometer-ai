import { AlertTriangle } from "lucide-react";

export default function RiskNotice({ warnings, error, health }) {
  const items = [
    "当前定标为候选/诊断定标，预测结果需要结合 ROI、profile 和实验条件判断。",
    ...(health?.warnings || []),
    ...(warnings || []),
  ];
  return (
    <section className="rounded-md border border-[#f2d4a1] bg-[#fff8eb] p-4">
      <h2 className="mb-2 flex items-center gap-2 text-[14px] font-semibold text-[#7a5200]">
        <AlertTriangle size={16} />
        风险提示
      </h2>
      {error ? <p className="mb-2 text-[13px] font-semibold text-lab-danger">错误原因：{error}</p> : null}
      <ul className="grid gap-1 text-[12px] text-[#7a5200]">
        {Array.from(new Set(items)).map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}
