function Generation({
  content,
  prob,
  isActive,
  isGenerationComplete,
}: {
  content: string;
  prob: number;
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
        <div className="Generation-finished">✓ Complete (Generated EOS)</div>
      )}
      {isActive && isGenerationComplete && (
        <div className="Generation-finished">
          ✓ Complete (Hit Max New Tokens)
        </div>
      )}
    </div>
  );
}

function getGenerationColor(prob: number) {
  return `rgba(247, 151, 141, ${prob * 0.8 + 0.2})`;
}

export default Generation;
