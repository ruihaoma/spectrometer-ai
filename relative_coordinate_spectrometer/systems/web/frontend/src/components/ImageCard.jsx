import { Image as ImageIcon, Loader2 } from "lucide-react";

export default function ImageCard({ title, imageUrl, placeholder, loading, stateLabel }) {
  return (
    <section className="chart-shell">
      <div className="chart-title">
        <span>{title}</span>
        {stateLabel ? (
          <span className="rounded border border-lab-line bg-lab-field px-2 py-0.5 text-[11px] font-medium text-lab-muted">
            {stateLabel}
          </span>
        ) : null}
      </div>
      <div className="chart-body flex items-center justify-center bg-[#fbfcfe]">
        {imageUrl ? (
          <img className="max-h-full max-w-full object-contain" src={imageUrl} alt={title} />
        ) : loading ? (
          <div className="flex flex-col items-center gap-2 text-center text-[13px] text-lab-muted">
            <Loader2 className="animate-spin text-lab-blue" size={26} />
            <span>正在处理图像</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-center text-[13px] text-lab-muted">
            <ImageIcon size={26} />
            <span>{placeholder}</span>
          </div>
        )}
      </div>
    </section>
  );
}
