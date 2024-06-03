import GenNode from "./GenNode";
import { LevelSpec } from "./types";

export default function GenLevel(level: LevelSpec) {
    return (
        <div className="GenLevel">
            {
                level.map(GenNode)
            }
        </div>
    );
}