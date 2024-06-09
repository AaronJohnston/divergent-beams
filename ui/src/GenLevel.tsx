import { useEffect, useMemo, useRef } from "react";
import GenNode from "./GenNode";
import { LevelSpec } from "./types";

export default function GenLevel({ level }: { level: LevelSpec }) {
  const nodesRef = useRef<HTMLElement[]>([]);

  const rendered = useMemo(() => {
    return (
      <div className="GenLevel">
        {level.nodes.map((node, idx) => {
          return (
            <GenNode
              key={idx}
              node={node}
              ref={(elem: HTMLElement) => (nodesRef.current[idx] = elem)}
            />
          );
        })}
      </div>
    );
  }, [level]);

  useEffect(() => {
    nodesRef.current = nodesRef.current.slice(0, level.nodes.length); // Make elements set by child renders visible as part of array
  }, [level, rendered]);

  useEffect(() => {
    if (nodesRef.current) {
      console.log("LEVEL EFFECT 2", level);
      for (let i = 0; i < level.nodes.length; i++) {
        const current = nodesRef.current[i];
        const node = level.nodes[i];
        if (
          current.parentElement &&
          current.parentElement.previousElementSibling &&
          node.parents !== undefined
        ) {
          for (const nodeParent of node.parents) {
            const parentElement =
              current.parentElement.previousElementSibling.children[nodeParent];
            drawEdge(parentElement, current);
          }
        }
      }
    }
  }, [level, rendered]);

  return rendered;
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

  // A margin of strokeWidth px is created around the SVG to show the line even in the horizontal or vertical case
  const strokeWidth = 1;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", (width + 2 * strokeWidth).toString());
  svg.setAttribute("height", (height + 2 * strokeWidth).toString());
  svg.setAttribute(
    "style",
    `position: absolute; top: ${
      svgTop + childRect.height / 2 - strokeWidth
    }px; left: ${-width - strokeWidth}px;`
  );

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", (linePoint1X + strokeWidth).toString());
  line.setAttribute("y1", (linePoint1Y + strokeWidth).toString());
  line.setAttribute("x2", (linePoint2X + strokeWidth).toString());
  line.setAttribute("y2", (linePoint2Y + strokeWidth).toString());
  line.setAttribute("stroke", "gray");
  line.setAttribute("stroke-width", strokeWidth.toString());

  svg.appendChild(line);

  if (child.lastElementChild && child.lastElementChild.tagName === "svg") {
    child.lastElementChild.remove();
  }

  child.appendChild(svg);
}
