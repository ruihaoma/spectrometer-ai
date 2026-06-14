import ActionPanel from "./ActionPanel.jsx";
import ExperimentInfoPanel from "./ExperimentInfoPanel.jsx";
import ExportPanel from "./ExportPanel.jsx";
import FileInputPanel from "./FileInputPanel.jsx";
import ModelInfoPanel from "./ModelInfoPanel.jsx";
import ProcessingConfigPanel from "./ProcessingConfigPanel.jsx";

export default function ControlPanel(props) {
  return (
    <aside className="border-r border-lab-line bg-white lg:w-[360px] lg:min-w-[360px] lg:overflow-y-auto">
      <FileInputPanel fileInfo={props.fileInfo} onFileChange={props.onFileChange} onClearFile={props.onClearFile} />
      <ActionPanel
        status={props.status}
        canPredict={props.canPredict}
        onPredict={props.onPredict}
        onClearResult={props.onClearResult}
      />
      <ExperimentInfoPanel
        sourceType={props.sourceType}
        exposureStatus={props.exposureStatus}
        note={props.note}
        onSourceType={props.onSourceType}
        onExposureStatus={props.onExposureStatus}
        onNote={props.onNote}
      />
      <ProcessingConfigPanel health={props.health} />
      <ModelInfoPanel health={props.health} />
      <ExportPanel hasResult={props.hasResult} onCsv={props.onCsv} onJson={props.onJson} onReport={props.onReport} />
    </aside>
  );
}
