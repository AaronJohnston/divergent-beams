import { useState } from "react";
import Switch from "react-switch";
import { PromptOptions } from "./types";

function PromptInput({
  evaluatePrompt,
}: {
  evaluatePrompt: (promptOptions: PromptOptions) => void;
}) {
  const [prompt, setPrompt] = useState(
    "What is the closest star to the Earth?"
  );
  const [maxBeams, setMaxBeams] = useState(5);
  const [maxNewTokens, setMaxNewTokens] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(5);
  const [topPDecayOn, setTopPDecayOn] = useState(false);
  const [topPDecay, setTopPDecay] = useState(0.95);
  const [gatherAlgo, setGatherAlgo] = useState<string>("farthest_neighbors");

  return (
    <div className="PromptInput">
      <textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        // onBlur={() => evaluatePrompt(currPrompt)}
      ></textarea>
      <div className="PromptInput-menu">
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
            value={topK}
            step={1}
            onChange={(e) => setTopK(parseFloat(e.target.value))}
          ></input>
          TOP-K
        </label>
        <label className="PromptInput-label">
          <Switch
            className="PromptInput-switch"
            checked={topPDecayOn}
            onChange={setTopPDecayOn}
            height={20}
            width={40}
            onColor="#7fd293"
            offColor="#aaa"
          ></Switch>
          TOP P DECAY
        </label>
        {topPDecayOn ? (
          <label className="PromptInput-label">
            <input
              className="PromptInput-number"
              type="number"
              value={topPDecay}
              onChange={(e) => setTopPDecay(parseFloat(e.target.value))}
            ></input>
            TOP P DECAY FACTOR
          </label>
        ) : (
          <></>
        )}
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
        <label className="PromptInput-label">
          <select
            value={gatherAlgo}
            onChange={(e) => setGatherAlgo(e.target.value)}
          >
            <option value="farthest_neighbors">kFN</option>
            <option value="k_means">K-Means</option>
          </select>
          GATHER ALGO
        </label>
        <button
          className="PromptInput-submit"
          onClick={() =>
            evaluatePrompt({
              prompt,
              topP,
              topK,
              maxBeams,
              maxNewTokens,
              topPDecay: topPDecayOn ? topPDecay : 1.0,
              gatherAlgo,
            })
          }
        >
          EVALUATE
        </button>
      </div>
    </div>
  );
}

export default PromptInput;
