function Generation({
  content,
  prob,
  finished,
}: {
  content: string;
  prob: number;
  finished: boolean;
}) {
  return (
    <div
      className="Generation"
      style={{ backgroundColor: getGenerationColor(prob) }}
    >
      <div className="Generation-content">{content}</div>
      {finished && <div className="Generation-finished">✓ Complete</div>}
    </div>
  );
}

function getGenerationColor(prob: number) {
  return `rgba(247, 151, 141, ${prob * 0.8 + 0.2})`;
}

export default Generation;
