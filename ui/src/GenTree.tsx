import GenLevel from "./GenLevel";
import { LevelSpec } from "./types";

export default function GenTree({
  levels,
  isGenerating,
  cancelGeneration,
}: {
  levels: LevelSpec[];
  isGenerating: boolean;
  cancelGeneration: () => void;
}) {
  return (
    <div className="GenTree">
      {isGenerating && (
        <div className="GenTree-generatingMenu">
          GENERATING...
          <button onClick={cancelGeneration}>CANCEL</button>
        </div>
      )}
      <div className="GenLevel">
        <div className="GenTree-prompt">PROMPT</div>
      </div>
      {levels.map((level) => (
        <GenLevel key={level.id} level={level} />
      ))}
    </div>
  );
}
