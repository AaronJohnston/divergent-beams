import { useState } from "react";
import Switch from "react-switch";
import { PromptOptions } from "./types";

function PromptInput({
  lastPromptOptions,
  evaluatePrompt,
}: {
  lastPromptOptions: PromptOptions | null;
  evaluatePrompt: (promptOptions: PromptOptions) => void;
}) {
  const [prompt, setPrompt] = useState(
    "What is the closest star to the Earth?"
  );
  const [embedPrune, setEmbedPrune] = useState(false);
  const [maxBeams, setMaxBeams] = useState(5);
  const [maxNewTokens, setMaxNewTokens] = useState(50);
  const [topP, setTopP] = useState(0.9);

  return (
    <div className="PromptInput">
      <textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
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
          PRUNE STEP
        </label>
        <label className="PromptInput-label">
          <input
            className="PromptInput-number"
            type="number"
            value={topP}
            step={0.01}
            onChange={(e) => setTopP(parseFloat(e.target.value))}
          ></input>
          TOP-P
        </label>
        <label className="PromptInput-label">
          <input
            className="PromptInput-number"
            type="number"
            value={maxBeams}
            onChange={(e) => setMaxBeams(parseInt(e.target.value))}
          ></input>
          MAX BEAMS
        </label>
        <label className="PromptInput-label">
          <input
            className="PromptInput-number"
            type="number"
            value={maxNewTokens}
            step={5}
            onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
          ></input>
          MAX NEW TOKENS
        </label>
        <button
          className="PromptInput-submit"
          onClick={() =>
            evaluatePrompt({ prompt, topP, maxBeams, maxNewTokens })
          }
        >
          EVALUATE
        </button>
      </div>
    </div>
  );
}

export default PromptInput;
