import { forwardRef } from "react";
import { NodeSpec } from "./types";

const GenNode = forwardRef(function GenNode({ node }: { node: NodeSpec }, ref) {
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
  return `rgba(232, 141, 103, ${node.prob * 0.7 + 0.3})`;
}
