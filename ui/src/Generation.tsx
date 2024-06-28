function Generation({
  content,
  prob,
  originalProb,
  isActive,
  isGenerationComplete,
}: {
  content: string;
  prob: number;
  originalProb: number;
  isActive: boolean;
  isGenerationComplete: boolean;
}) {
  return (
    <div
      className="Generation"
      style={{ backgroundColor: getGenerationColor(prob) }}
    >
      <div className="Generation-content">{content}</div>
      {!isActive && (
        <div className="Generation-finished">✓ Complete (Hit EOS)</div>
      )}
      {isActive && isGenerationComplete && (
        <div className="Generation-finished">
          ✓ Complete (Hit Max New Tokens)
        </div>
      )}
      <div className="Generation-prob">
        logprob: {originalProb.toPrecision(3)}
      </div>
    </div>
  );
}

function getGenerationColor(prob: number) {
  return `hsla(6, 87%, ${Math.round(94 - prob * 18)}%, 1)`;
}

export default Generation;
