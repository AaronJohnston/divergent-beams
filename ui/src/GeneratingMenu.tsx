function GeneratingMenu({
  isGenerating,
  cancelGeneration,
}: {
  isGenerating: boolean;
  cancelGeneration: () => void;
}) {
  return (
    <div className="GeneratingMenu">
      {isGenerating ? (
        <button className="GeneratingMenu-cancel" onClick={cancelGeneration}>
          Cancel Generation
        </button>
      ) : null}
    </div>
  );
}

export default GeneratingMenu;
