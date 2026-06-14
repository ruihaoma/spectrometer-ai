export default function LogPanel({ logs }) {
  const content = logs?.length ? logs : ["等待上传图片", "等待 ROI 提取", "等待 profile 生成", "等待模型预测"];
  return (
    <section className="rounded-md border border-lab-line bg-[#101828] p-4 text-[#d0d5dd]">
      <h2 className="mb-2 text-[14px] font-semibold text-white">处理日志</h2>
      <ol className="grid max-h-32 gap-1 overflow-auto text-[12px]">
        {content.map((item, index) => (
          <li key={`${item}-${index}`} className="font-mono">
            {String(index + 1).padStart(2, "0")}  {item}
          </li>
        ))}
      </ol>
    </section>
  );
}
