import { forwardRef } from "react";
import { NodeSpec } from "./types";

const GenNode = forwardRef<
  HTMLDivElement,
  { node: NodeSpec; totalProb: number }
>(function GenNode(
  { node, totalProb }: { node: NodeSpec; totalProb: number },
  ref
) {
  return (
    <div
      className="GenNode"
      ref={ref}
      style={{ backgroundColor: getNodeColor(node, totalProb) }}
    >
      {node.content.replace(/▁/g, "").replace(/<0x0A>/g, "\\n")}
    </div>
  );
});

export default GenNode;

function getNodeColor(node: NodeSpec, totalProb: number) {
  return `rgba(247, 151, 141, ${(node.prob / totalProb) * 0.8 + 0.2})`;
}
