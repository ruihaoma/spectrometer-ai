import { formatNumber } from "../utils/format.js";

export default function ResultSummary({ result, status, sourceType, exposureStatus }) {
  return (
    <section className="rounded-md border border-lab-line bg-white p-4">
      <h2 className="mb-3 text-[14px] font-semibold text-lab-ink">推理结果摘要</h2>
      <dl className="grid gap-2 text-[12px] sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <dt className="text-lab-muted">预测状态</dt>
          <dd className="font-semibold text-lab-ink">{status}</dd>
        </div>
        <div>
          <dt className="text-lab-muted">主峰波长</dt>
          <dd className="font-semibold text-lab-ink">{formatNumber(result?.prediction?.peak_wavelength_nm, 3)} nm</dd>
        </div>
        <div>
          <dt className="text-lab-muted">峰值强度</dt>
          <dd className="font-semibold text-lab-ink">{formatNumber(result?.prediction?.peak_intensity, 6)}</dd>
        </div>
        <div>
          <dt className="text-lab-muted">波长范围</dt>
          <dd className="font-semibold text-lab-ink">400-650 nm</dd>
        </div>
        <div>
          <dt className="text-lab-muted">输入 shape</dt>
          <dd className="font-semibold text-lab-ink">[1, 4, 2501]</dd>
        </div>
        <div>
          <dt className="text-lab-muted">输出 shape</dt>
          <dd className="font-semibold text-lab-ink">[1, 2501]</dd>
        </div>
        <div>
          <dt className="text-lab-muted">模型名称</dt>
          <dd className="font-semibold text-lab-ink">spectrum_unet_transformer_1d</dd>
        </div>
        <div>
          <dt className="text-lab-muted">定标版本</dt>
          <dd className="break-all font-semibold text-lab-ink">
            {result?.processing_info?.calibration_version || "relative_spectral_coordinate_linear_diagnostic"}
          </dd>
        </div>
        <div>
          <dt className="text-lab-muted">光源类型</dt>
          <dd className="font-semibold text-lab-ink">{result?.experiment_info?.source_type || sourceType}</dd>
        </div>
        <div>
          <dt className="text-lab-muted">曝光状态</dt>
          <dd className="font-semibold text-lab-ink">{result?.experiment_info?.exposure_status || exposureStatus}</dd>
        </div>
      </dl>
    </section>
  );
}
