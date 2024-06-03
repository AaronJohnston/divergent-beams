import { useEffect, useRef } from "react";
import GenNode from "./GenNode";
import { LevelSpec } from "./types";

export default function GenLevel({
    level
}: {
    level: LevelSpec
}) {

    const nodesRef = useRef<HTMLElement[]>([]);

    useEffect(() => {
        nodesRef.current = nodesRef.current.slice(0, level.length); // Make elements set by child renders visible as part of array
    }, [level]);

    useEffect(() => {
        if (nodesRef.current) {
            for (let i = 0; i < level.length; i++) {
                const current = nodesRef.current[i];
                const node = level[i];
                if (current.parentElement && current.parentElement.previousElementSibling && node.parent !== undefined) {
                    const parent = current.parentElement.previousElementSibling.children[node.parent];
                    drawEdge(parent, current);
                }
            }
        }
    }, [level]);

    return (
        <div className="GenLevel">
            {
                level.map((node, idx) => {
                    return (
                        <GenNode key={node.content} node={node} ref={(elem: HTMLElement) => nodesRef.current[idx] = elem} />
                    );
                })
            }
        </div>
    );
}

function drawEdge(parent: Element, child: Element) {
    const parentRect = parent.getBoundingClientRect();
    const childRect = child.getBoundingClientRect();

    const parentPointX = parentRect.left + parentRect.width;
    const parentPointY = parentRect.top + parentRect.height / 2;

    const childPointX = childRect.left;
    const childPointY = childRect.top + childRect.height / 2;

    const width = Math.abs(childPointX - parentPointX);
    const height = Math.abs(childPointY - parentPointY);

    const linePoint1X = 0;
    const linePoint1Y = parentPointY < childPointY ? 0 : height;

    const linePoint2X = width;
    const linePoint2Y = parentPointY < childPointY ? height : 0;

    const svgTop = parentPointY < childPointY ? -height : 0;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', (width + 2).toString()); // + 2 to show line in vertical case
    svg.setAttribute('height', (height + 2).toString()); // + 2 to show line in horizontal case
    svg.setAttribute('style', `position: absolute; top: ${svgTop + parentRect.height / 2}px; left: ${-width}px;`);

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', (linePoint1X + 1).toString());
    line.setAttribute('y1', (linePoint1Y + 1).toString());
    line.setAttribute('x2', (linePoint2X + 1).toString());
    line.setAttribute('y2', (linePoint2Y + 1).toString());
    line.setAttribute('stroke', 'gray');
    line.setAttribute('stroke-width', '1');

    svg.appendChild(line);

    if (child.lastElementChild && child.lastElementChild.tagName === 'svg') {
        child.lastElementChild.remove();
    }

    child.appendChild(svg);
}