import { forwardRef, useState } from "react";
import { NodeSpec } from "./types";

const GenNode = forwardRef<HTMLDivElement, { node: NodeSpec }>(function GenNode(
  { node }: { node: NodeSpec },
  ref
) {
  // const [isOpen, setIsOpen] = useState(false);

  let displayContent = node.content.replace(/▁/g, "").replace(/<0x0A>/g, "\\n");
  if (displayContent === "") {
    displayContent = "<SPECIAL>";
  }

  return (
    <div className="GenNode" ref={ref}>
      <div
        className="GenNode-content"
        style={{ backgroundColor: getNodeColor(node) }}
        // onMouseEnter={() => setIsOpen(true)}
        // onClick={() => setIsOpen(true)}
        // onMouseLeave={() => setIsOpen(false)}
        // onBlur={() => setIsOpen(false)}
      >
        {displayContent}
      </div>
      {/* {isOpen && (
        <div
          className="GenNode-prob"
          style={{ backgroundColor: getNodeColor(node) }}
        >
          logprob: {node.prob.toPrecision(3)}
        </div>
      )} */}
    </div>
  );
});

export default GenNode;

function getNodeColor(node: NodeSpec) {
  return `hsla(6, 87%, ${Math.round(94 - node.prob * 18)}%, 1)`;
}
