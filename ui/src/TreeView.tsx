import './TreeView.css';

export type Level = Node[];

export type Node = {
    content: string;
    prob: number;
    parent?: number;
    status?: string[];
};

export default function TreeView({
    levels
}: {
    levels: Level[]
}) {
    return (
        <div className="TreeView">
            {
                levels.map(renderLevel)
            }
        </div>
    );
}

function renderLevel(level: Level) {
    return (
        <div className="TreeView-Level">
            {
                level.map(renderNode)
            }
        </div>
    );
}

function renderNode(node: Node) {
    return (
        <div className="TreeView-Node">
            {node.content} ({node.prob})
        </div>
    );
}