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
                levels.map(GenLevel)
            }
        </div>
    );
}