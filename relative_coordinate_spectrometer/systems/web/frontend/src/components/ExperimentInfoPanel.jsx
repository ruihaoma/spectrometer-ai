import { FlaskConical } from "lucide-react";

const sourceOptions = ["未知", "白光 LED", "红 LED", "绿 LED", "蓝 LED", "紫色 LED", "黄色 LED", "Hg", "Na", "He-Ne", "其他"];
const exposureOptions = ["自动", "正常", "偏暗", "过曝"];

export default function ExperimentInfoPanel({ sourceType, exposureStatus, note, onSourceType, onExposureStatus, onNote }) {
  return (
    <section className="tool-panel">
      <h2 className="panel-title">
        <FlaskConical size={15} />
        实验信息区
      </h2>
      <div className="grid gap-2">
        <label>
          <span className="field-label">光源类型</span>
          <select className="field-value w-full" value={sourceType} onChange={(event) => onSourceType(event.target.value)}>
            {sourceOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="field-label">曝光状态</span>
          <select
            className="field-value w-full"
            value={exposureStatus}
            onChange={(event) => onExposureStatus(event.target.value)}
          >
            {exposureOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="field-label">样本备注</span>
          <textarea
            className="field-value min-h-16 w-full resize-none"
            value={note}
            onChange={(event) => onNote(event.target.value)}
            placeholder="可填写样本批次、拍摄条件或观察备注"
          />
        </label>
      </div>
    </section>
  );
}
