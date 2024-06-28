import GenLevel from "./GenLevel";
import { LevelSpec } from "./types";

export default function GenTree({ levels }: { levels: LevelSpec[] }) {
  return (
    <div className="GenTree horiz-scroll horiz-scroll-reverse">
      <div className="horiz-scroll-reverse-content">
        <div className="GenLevel">
          <div className="GenLevel-label"></div>
          <div className="GenLevel-timing"></div>
          <div className="GenLevel-nodes">
            <div
              className={`GenTree-prompt ${levels.length > 0 ? "" : "hidden"}`}
            >
              PROMPT
            </div>
          </div>
        </div>
        {levels.map((level) => (
          <GenLevel key={level.id} level={level} />
        ))}
      </div>
    </div>
  );
}
