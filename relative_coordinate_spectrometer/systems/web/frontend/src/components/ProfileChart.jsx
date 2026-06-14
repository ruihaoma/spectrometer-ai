import { useMemo } from "react";
import { Loader2 } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function buildData(result) {
  if (!result) return [];
  const wavelength = result.wavelength_nm || [];
  const profile = result.profile || {};
  const stride = Math.max(1, Math.floor(wavelength.length / 650));
  return wavelength
    .map((nm, index) => ({
      nm,
      R: profile.R?.[index],
      G: profile.G?.[index],
      B: profile.B?.[index],
      Gray: profile.Gray?.[index],
    }))
    .filter((_, index) => index % stride === 0);
}

export default function ProfileChart({ result, status }) {
  const data = useMemo(() => buildData(result), [result]);
  const isProcessing = status === "处理中";
  return (
    <section className="chart-shell">
      <div className="chart-title">
        <span>R/G/B/Gray 四通道输入</span>
        <span className="rounded border border-lab-line bg-lab-field px-2 py-0.5 text-[11px] font-medium text-lab-muted">
          {data.length ? "已生成" : isProcessing ? "处理中" : "等待"}
        </span>
      </div>
      <div className="chart-body">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 2 }}>
              <CartesianGrid stroke="#e4e9f1" strokeDasharray="3 3" />
              <XAxis dataKey="nm" type="number" domain={[400, 650]} tick={{ fontSize: 11 }} tickCount={6} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} width={34} />
              <Tooltip formatter={(value) => Number(value).toFixed(4)} labelFormatter={(value) => `${value} nm`} />
              <Line type="monotone" dataKey="R" stroke="#d92d20" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="G" stroke="#12b76a" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="B" stroke="#1570ef" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="Gray" stroke="#475467" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : isProcessing ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-[13px] text-lab-muted">
            <Loader2 className="animate-spin text-lab-blue" size={24} />
            <span>正在生成四通道 profile</span>
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-[13px] text-lab-muted">等待 profile 生成</div>
        )}
      </div>
    </section>
  );
}
