import { useState } from "react";
import "./App.css";
import GenTree from "./GenTree";
import PromptInput from "./PromptInput";
import { FinishedSpec, LevelSpec, PromptOptions } from "./types";
import Generations from "./Generations";
import GeneratingMenu from "./GeneratingMenu";

const TREE_ENDPOINT =
  "http://ec2-52-89-34-232.us-west-2.compute.amazonaws.com/api/v1/tree";

function App() {
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [levels, setLevels] = useState<LevelSpec[]>([]);
  const [finished, setFinished] = useState<FinishedSpec[]>([]);
  const [isError, setIsError] = useState<boolean>(false);

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

    eventSource.onerror = (event) => {
      console.error("EventSource error:", event);
      setIsError(true);
      eventSource.close();
    };

    eventSource.onmessage = (event) => {
      if (event.lastEventId === "END") {
        console.log("CLOSING EVENT SOURCE");
        eventSource.close();
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
        <Generations levels={levels} finished={finished}></Generations>
      </div>
    </div>
  );
}

export default App;
