import { useEffect, useState } from 'react';
import GenLevel from './GenLevel';
import './GenTree.css';
import { LevelSpec } from './types';

const TREE_ENDPOINT = 'http://localhost:8000/api/v1/tree';

export default function GenTree({
    prompt
}: {
    prompt: string;
}) {
    const [levels, setLevels] = useState<LevelSpec[]>([]);

    console.log('RENDERING GENTREE', levels.toString());

    useEffect(() => {
        console.log('OPENING EVENT SOURCE');
        const eventSource = new EventSource(TREE_ENDPOINT + '?prompt=' + prompt);
        setLevels([]);

        eventSource.onerror = (event) => {
            console.error('EventSource failed:', event);
            eventSource.close();
        }

        eventSource.addEventListener('level', (event) => {
            if (event.lastEventId === 'END') {
                console.log('CLOSING EVENT SOURCE');
                eventSource.close();
                return;
            }
            console.log(event);
            const data: LevelSpec = JSON.parse(event.data);
            setLevels((levels: LevelSpec[]) => [...levels, data]);
        });

        return () => {
            console.log('CLEANING UP EVENT SOURCE');
            eventSource.close();
        }
    }, [prompt]);

    return (
        <div className="GenTree">
            <div className="GenPromptLevel">
                <div className="GenPrompt">
                    <div className="GenPromptTokens">
                        {prompt}
                    </div>
                    <div className="GenPromptMenu">
                        <p>EDIT PROMPT</p>
                    </div>
                </div>
            </div>
            {
                levels.map((level, idx) => <GenLevel key={idx} level={level} />)
            }
        </div>
    );
}