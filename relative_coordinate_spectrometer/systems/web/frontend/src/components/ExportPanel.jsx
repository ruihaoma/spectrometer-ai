import { Download } from "lucide-react";

export default function ExportPanel({ hasResult, onCsv, onJson, onReport }) {
  return (
    <section className="tool-panel">
      <h2 className="panel-title">
        <Download size={15} />
        导出工具区
      </h2>
      <div className="grid gap-2">
        <button className="control-button justify-start" type="button" onClick={onCsv} disabled={!hasResult}>
          <Download size={15} />
          导出 prediction.csv
        </button>
        <button className="control-button justify-start" type="button" onClick={onJson} disabled={!hasResult}>
          <Download size={15} />
          导出 prediction.json
        </button>
        <button className="control-button justify-start" type="button" onClick={onReport} disabled={!hasResult}>
          <Download size={15} />
          导出处理报告
        </button>
      </div>
    </section>
  );
}
