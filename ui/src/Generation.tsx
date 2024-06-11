function Generation({ content, prob }: { content: string; prob: number }) {
  return (
    <div className="Generation">
      <div className="Generation-prob">{prob.toFixed(2)}</div>
      <div className="Generation-content">{content}</div>
    </div>
  );
}

export default Generation;
