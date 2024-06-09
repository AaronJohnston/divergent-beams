import GenLevel from "./GenLevel";
import { LevelSpec } from "./types";

export default function GenTree({ levels }: { levels: LevelSpec[] }) {
  console.log("RENDERING GENTREE", levels.toString());

  return (
    <div className="GenTree">
      <div className="GenLevel">
        <div className="GenTree-prompt">PROMPT</div>
      </div>
      {levels.map((level, idx) => (
        <GenLevel key={idx} level={level} />
      ))}
    </div>
  );
}
