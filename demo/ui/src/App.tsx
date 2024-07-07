import { useState } from "react";
import "./App.css";
import GenTree from "./GenTree";
import PromptInput from "./PromptInput";
import { FinishedSpec, LevelSpec, PromptOptions } from "./types";
import Generations from "./Generations";
import GeneratingMenu from "./GeneratingMenu";
import Hint from "./Hint";

const TREE_ENDPOINT =
  "https://ec2-52-89-34-232.us-west-2.compute.amazonaws.com/api/v1/tree";

function App() {
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [levels, setLevels] = useState<LevelSpec[]>([]);
  const [finished, setFinished] = useState<FinishedSpec[]>([]);
  const [isError, setIsError] = useState<boolean>(false);
  const [isGenerationComplete, setIsGenerationComplete] =
    useState<boolean>(false);

  const cancelGeneration = () => {
    if (eventSource) {
      eventSource.close();
    }
    setEventSource(() => null);
  };

  const evaluatePrompt = (promptOptions: PromptOptions) => {
    cancelGeneration();
    console.log("OPENING EVENT SOURCE");
    const url = `${TREE_ENDPOINT}?topP=${promptOptions.topP}&topPDecay=${promptOptions.topPDecay}&topK=${promptOptions.topK}&maxBeams=${promptOptions.maxBeams}&maxNewTokens=${promptOptions.maxNewTokens}&gatherAlgo=${promptOptions.gatherAlgo}&prompt=${promptOptions.prompt}`;
    const eventSource = new EventSource(url);
    setLevels([]);
    setFinished([]);
    setIsGenerationComplete(false);

    eventSource.onerror = (event) => {
      console.error("EventSource error:", event);
      setIsError(true);
      eventSource.close();
    };

    eventSource.onmessage = (event) => {
      if (event.lastEventId === "END") {
        console.log("CLOSING EVENT SOURCE");
        eventSource.close();
        setIsGenerationComplete(true);
        return;
      }
      const data: LevelSpec = JSON.parse(event.data);
      setLevels((levels: LevelSpec[]) => [...levels, data]);
      if (data.finished) {
        setFinished((finished: FinishedSpec[]) => [
          ...finished,
          ...data.finished,
        ]);
      }
    };

    setEventSource(eventSource);
  };

  return (
    <div className="App-root">
      <header className="App-header">
        <h1>Divergent Beams</h1>
        <Hint width="500px" onRight={true}>
          <h3>Divergent Beams</h3>
          <p>
            A tool for exploring the possible outputs of a language model. While
            most language model sampling strategies aim to produce the most{" "}
            <i>likely</i> output(s), Divergent Beams aims to produce the most{" "}
            <i>diverse</i> outputs. It does this by sending a set of beams
            through the model's probability space, selecting completions that
            are probable (controlled by Top-P and Top-K) but as different from
            one another as possible (controlled by the gather algorithm). Each
            beam keeps track of the joint probability of its entire sequence.
            For each token, the algorithm runs at most 2 steps:
          </p>
          <h4>Sample Step</h4>
          <p>
            For every active beam, some number of likely next tokens are sampled
            according to Top-P and Top-K. These tokens become new beams. The
            underlying algorithm takes advantage of GPU concurrency by doing
            this sampling in a batch size of 8.
          </p>
          <h4>Gather Step</h4>
          <p>
            When the Sample step produces more beams than Max Beams allows, the
            beams are gathered into the "most different" representatives, where
            "different" is defined as distance in the model's latent space. The
            underlying algorithm has implementations of k-Farthest Neighbors or
            K-Means for selecting representatives, and the probability mass of
            each beam is consolidated into its closest representative.
          </p>

          <br />
          <p>
            Each beam terminates when it hits the EOS token or the Max New
            Tokens limit. Currently only Microsoft's Phi-3-mini-4k-instruct
            model is supported.
          </p>
        </Hint>
        <div className="Model-select">
          <label>
            Model
            <select>
              <option value="microsoft/phi-3-mini-4k-instruct">
                microsoft/phi-3-mini-4k-instruct
              </option>
            </select>
          </label>
        </div>
      </header>
      <div className="App-content">
        <PromptInput evaluatePrompt={evaluatePrompt}></PromptInput>
        <GeneratingMenu
          isError={isError}
          isGenerating={
            !!eventSource && eventSource.readyState === eventSource.OPEN
          }
          cancelGeneration={cancelGeneration}
        ></GeneratingMenu>
        <GenTree levels={levels}></GenTree>
        <Generations
          levels={levels}
          finished={finished}
          isGenerationComplete={isGenerationComplete}
        ></Generations>
      </div>
    </div>
  );
}

export default App;
