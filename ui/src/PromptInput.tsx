import { useState } from "react";
import Switch from "react-switch";

function PromptInput({
  evaluatePrompt,
}: {
  evaluatePrompt: (prompt: string) => void;
}) {
  const [currPrompt, setCurrPrompt] = useState(
    "What is the closest star to the Earth?"
  );
  const [embedPrune, setEmbedPrune] = useState(false);

  return (
    <div className="PromptInput">
      <textarea
        value={currPrompt}
        onChange={(event) => setCurrPrompt(event.target.value)}
        // onBlur={() => evaluatePrompt(currPrompt)}
      ></textarea>
      <div className="PromptInput-menu">
        <label className="PromptInput-label">
          <Switch
            className="PromptInput-switch"
            checked={embedPrune}
            onChange={setEmbedPrune}
            height={20}
            width={40}
            onColor="#7fd293"
            offColor="#aaa"
          ></Switch>
          PRUNE
        </label>
        <button
          className="PromptInput-submit"
          onClick={() => evaluatePrompt(currPrompt)}
        >
          EVALUATE
        </button>
      </div>
    </div>
  );
}

export default PromptInput;
