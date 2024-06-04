import { useEffect, useState } from 'react';
import GenLevel from './GenLevel';
import './GenTree.css';
import { LevelSpec } from './types';

const TREE_ENDPOINT = 'http://localhost:8000/api/v1/tree'

const levels = [
    [
      { 'content': 'Mir', 'prob': 0.4 },
      { 'content': 'Cat', 'prob': 0.2 },
      { 'content': 'Bog', 'prob': 0.2 },
    ],
    [
      { 'content': 'ror', 'prob': 0.4, 'parent': 0 },
      { 'content': 'dog', 'prob': 0.4, 'parent': 0 },
      { 'content': 'ture', 'prob': 0.4, 'parent': 1  },
      { 'content': 'sap', 'prob': 0.4, 'parent': 1, 'status': ['eliminated'] },
      { 'content': 'lant', 'prob': 0.4, 'parent': 2  },
    ],
    [
      { 'content': 'ror', 'prob': 0.4, 'parent': 0 },
      { 'content': 'dog', 'prob': 0.4, 'parent': 1 },
      { 'content': 'ture', 'prob': 0.4, 'parent': 2  },
      { 'content': 'sap', 'prob': 0.4, 'parent': 2  },
      { 'content': 'snap', 'prob': 0.4, 'parent': 4  },
      { 'content': 'lant', 'prob': 0.4, 'parent': 4  },
    ]
  ]


export default function GenTree({
    prompt
}: {
    prompt: string;
}) {
    const [levels, setLevels] = useState<LevelSpec[]>([]);

    useEffect(() => {
        const eventSource = new EventSource(TREE_ENDPOINT + '?prompt=' + prompt);
        eventSource.onmessage = (event) => {
            const data: LevelSpec = JSON.parse(event.data);
            setLevels((levels: LevelSpec[]) => [...levels, data]);
        };
    }, [prompt]);

    return (
        <div className="GenTree">
            {
                levels.map((level, idx) => <GenLevel key={idx} level={level} />)
            }
        </div>
    );
}