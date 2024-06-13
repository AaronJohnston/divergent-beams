function GeneratingMenu({
  isError,
  isGenerating,
  cancelGeneration,
}: {
  isError: boolean;
  isGenerating: boolean;
  cancelGeneration: () => void;
}) {
  return (
    <div className="GeneratingMenu">
      {isError ? (
        <>
          <span className="GeneratingMenu-error">
            ERROR CONNECTING TO SERVER! PLEASE TRY AGAIN LATER.
          </span>
        </>
      ) : isGenerating ? (
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
