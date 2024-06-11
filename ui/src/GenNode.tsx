import { forwardRef } from "react";
import { NodeSpec } from "./types";

const GenNode = forwardRef<HTMLDivElement, { node: NodeSpec }>(function GenNode(
  { node }: { node: NodeSpec },
  ref
) {
  return (
    <div
      className="GenNode"
      ref={ref}
      style={{ backgroundColor: getNodeColor(node) }}
    >
      {node.content.replace(/▁/g, "")}
    </div>
  );
});

export default GenNode;

function getNodeColor(node: NodeSpec) {
  return `rgba(247, 151, 141, ${node.prob * 0.8 + 0.2})`;
}
