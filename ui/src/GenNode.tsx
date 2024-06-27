import { forwardRef } from "react";
import { NodeSpec } from "./types";

const GenNode = forwardRef<HTMLDivElement, { node: NodeSpec }>(function GenNode(
  { node }: { node: NodeSpec },
  ref
) {
  let displayContent = node.content.replace(/▁/g, "").replace(/<0x0A>/g, "\\n");
  if (displayContent === "") {
    displayContent = "<SPECIAL>";
  }

  return (
    <div className="GenNode" ref={ref}>
      <div
        className="GenNode-content"
        style={{ backgroundColor: getNodeColor(node) }}
      >
        {displayContent}
      </div>
    </div>
  );
});

export default GenNode;

function getNodeColor(node: NodeSpec) {
  return `rgba(247, 151, 141, ${node.prob * 0.8 + 0.2})`;
}
