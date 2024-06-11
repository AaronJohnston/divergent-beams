function Generation({ content, prob }: { content: string; prob: number }) {
  return (
    <div
      className="Generation"
      style={{ backgroundColor: getGenerationColor(prob) }}
    >
      <div className="Generation-prob">{prob.toFixed(2)}</div>
      <div className="Generation-content">{content}</div>
    </div>
  );
}

function getGenerationColor(prob: number) {
  return `rgba(247, 151, 141, ${prob * 0.8 + 0.2})`;
}

export default Generation;
