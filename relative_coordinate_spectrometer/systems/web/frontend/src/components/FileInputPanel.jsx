import { FileImage, Trash2, Upload } from "lucide-react";
import { fileFormat } from "../utils/format.js";

export default function FileInputPanel({ fileInfo, onFileChange, onClearFile }) {
  return (
    <section className="tool-panel">
      <h2 className="panel-title">
        <FileImage size={15} />
        数据输入区
      </h2>
      <label className="primary-button w-full cursor-pointer" htmlFor="spectrum-file">
        <Upload size={15} />
        选择 DIY 光谱照片
      </label>
      <input
        id="spectrum-file"
        className="hidden"
        type="file"
        accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,image/png,image/jpeg,image/bmp,image/tiff"
        onChange={(event) => {
          onFileChange(event.target.files?.[0] || null);
          event.target.value = "";
        }}
      />
      <div className="mt-3 grid gap-2">
        <div>
          <span className="field-label">文件名</span>
          <div className="field-value break-all">{fileInfo?.name || "未选择"}</div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="field-label">图像尺寸</span>
            <div className="field-value">{fileInfo ? `${fileInfo.width} × ${fileInfo.height}` : "--"}</div>
          </div>
          <div>
            <span className="field-label">文件格式</span>
            <div className="field-value">{fileInfo ? fileFormat(fileInfo.name) : "--"}</div>
          </div>
        </div>
      </div>
      <button className="control-button mt-3 w-full" type="button" onClick={onClearFile} disabled={!fileInfo}>
        <Trash2 size={15} />
        清空当前文件
      </button>
    </section>
  );
}
