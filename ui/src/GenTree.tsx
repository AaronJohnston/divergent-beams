import GenLevel from './GenLevel';
import './GenTree.css';
import { LevelSpec } from './types';

export default function GenTree({
    levels
}: {
    levels: LevelSpec[]
}) {
    return (
        <div className="GenTree">
            {
                levels.map((level, idx) => <GenLevel key={idx} level={level} />)
            }
        </div>
    );
}