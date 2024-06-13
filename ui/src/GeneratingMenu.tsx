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
        <>
          <span className="loader"></span>
          <p>GENERATING...</p>
          <button className="GeneratingMenu-cancel" onClick={cancelGeneration}>
            CANCEL
          </button>
        </>
      ) : null}
    </div>
  );
}

export default GeneratingMenu;
