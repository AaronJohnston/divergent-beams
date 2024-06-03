import { useEffect, useRef } from "react";
import { NodeSpec } from "./types";

export default function GenNode(node: NodeSpec) {

    const ref = useRef(null);

    useEffect(() => {
        if (ref.current) {
            const current: HTMLElement = ref.current;
            if (current.parentElement && current.parentElement.previousElementSibling && node.parent) {
                const parent = current.parentElement.previousElementSibling.children[node.parent];
                drawEdge(parent, current)
            }
        } else {
            console.log('no ref');
        }
    }, [node.parent]);

    return (
        <div className="GenNode" ref={ref} style={{backgroundColor: getNodeColor(node)}}>
            {node.content}
        </div>
    );
}

function getNodeColor(node: NodeSpec) {
    return `rgba(223, 120, 97, ${node.prob * 0.7 + 0.3})`;
}

function drawEdge(parent: Element, child: Element) {

}