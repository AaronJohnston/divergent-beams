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
        <div>
        <h1>Tree View</h1>
        </div>
    );
}