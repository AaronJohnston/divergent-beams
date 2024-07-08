import { useState } from "react";
import Switch from "react-switch";
import { PromptOptions } from "./types";
import Hint from "./Hint";

function PromptInput({
  evaluatePrompt,
}: {
  evaluatePrompt: (promptOptions: PromptOptions) => void;
}) {
  const [prompt, setPrompt] = useState(
    "Can you write a CRON string to run every 5 minutes and also at 2:03pm every day? If possible, write the string with no explanation. If not, explain why in 1 sentence."
  );
  const [maxBeams, setMaxBeams] = useState(5);
  const [maxNewTokens, setMaxNewTokens] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(5);
  const [topPDecayOn, setTopPDecayOn] = useState(true);
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
          <Hint>
            <h3>TOP-P</h3>
            Top-P controls how much probability mass to sample for each beam at
            each step. A Top-P of 0.9 means top tokens will be selected until
            they represent 90% of the probability mass. Works with Top-K by
            choosing the most restrictive limit (whichever produces the fewest
            tokens at a given step).
          </Hint>
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
          <Hint>
            <h3>TOP-K</h3>
            Top-K controls the maximum number of tokens that can be sampled for
            each beam at each step. A Top-K of 3 means at most the 3 top tokens
            will be selected. Works with Top-P by choosing the most restrictive
            limit (whichever produces the fewest tokens at a given step).
          </Hint>
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
          TOP-P DECAY
          <Hint>
            <h3>TOP-P DECAY</h3>
            If enabled, the Top-P value will decay at each Sample step of the
            algorithm. This can help with diversity of the generations by
            favoring divergent choices earlier in the text, and speeds up the
            algorithm by preventing the tree from exploding in size.
          </Hint>
        </label>
        {topPDecayOn ? (
          <label className="PromptInput-label">
            <input
              className="PromptInput-number"
              type="number"
              value={topPDecay}
              step={(0.01).toFixed(2)}
              onChange={(e) => setTopPDecay(parseFloat(e.target.value))}
            ></input>
            TOP-P DECAY FACTOR
            <Hint>
              <h3>TOP-P DECAY FACTOR</h3>
              The factor to decay Top-P by. At each Sample step of the
              algorithm, after sampling for all active beams, Top-P will be
              multiplied by Top-P Decay Factor. A value of 0.99 means Top-P will
              decay by 1% at each step. A value of 1.0 means Top-P will not
              decay.
            </Hint>
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
          <Hint>
            <h3>MAX BEAMS</h3>
            The max number of beams to keep active after each new token is
            generated. If a Sample step generates more than Max Beams, a Gather
            step will reduce their number to keep only the most diverse. A
            higher number of beams will increase the diversity of the
            generations, but will also slow down the algorithm.
          </Hint>
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
          <Hint>
            <h3>MAX NEW TOKENS</h3>
            The max number of new tokens that can be generated in the algorithm.
          </Hint>
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
          <Hint>
            <h3>GATHER ALGO</h3>
            What algorithm to use during a Gather step to select the most
            diverse beams. kFN (K-Farthest Neighbors) will select those that are
            the farthest from each other, based on cosine distance between last
            hidden states. K-Means will cluster the beams based on last hidden
            states and select the most central ones. kFN is faster.
          </Hint>
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
