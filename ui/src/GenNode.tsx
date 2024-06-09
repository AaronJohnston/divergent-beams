import { forwardRef } from "react";
import { NodeSpec } from "./types";

const GenNode = forwardRef<HTMLDivElement, { node: NodeSpec }>(function GenNode(
  { node }: { node: NodeSpec },
  ref
) {
  // console.log("RENDERING NODE", node);
  return (
    <div
      className="GenNode"
      ref={ref}
      style={{ backgroundColor: getNodeColor(node) }}
    >
      {node.content}
    </div>
  );
});

export default GenNode;

function getNodeColor(node: NodeSpec) {
  return `rgba(232, 141, 103, ${(node.prob + 10) / 10})`;
}
