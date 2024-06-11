function Generation({
  content,
  prob,
  totalProb,
}: {
  content: string;
  prob: number;
  totalProb: number;
}) {
  return (
    <div
      className="Generation"
      style={{ backgroundColor: getGenerationColor(prob, totalProb) }}
    >
      <div className="Generation-prob">{prob.toFixed(2)}</div>
      <div className="Generation-content">{content}</div>
    </div>
  );
}

function getGenerationColor(prob: number, totalProb: number) {
  return `rgba(247, 151, 141, ${(prob / totalProb) * 0.8 + 0.2})`;
}

export default Generation;
