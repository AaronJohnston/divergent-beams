import { NodeSpec } from "./types";

export default function GenNode(node: NodeSpec) {
    return (
        <div className="GenNode" style={{backgroundColor: getNodeColor(node)}}>
            {node.content}
        </div>
    );
}

function getNodeColor(node: NodeSpec) {
    return `rgba(223, 120, 97, ${node.prob * 0.7 + 0.3})`;
}