import { useState } from "react";
import "./App.css";
import GenTree from "./GenTree";
import PromptInput from "./PromptInput";
import { LevelSpec, PromptOptions } from "./types";
import Generations from "./Generations";

const TREE_ENDPOINT =
  "http://ec2-52-89-34-232.us-west-2.compute.amazonaws.com/api/v1/tree";

function App() {
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [levels, setLevels] = useState<LevelSpec[]>([]);
  const [lastPromptOptions, setLastPromptOptions] =
    useState<PromptOptions | null>(null);

  const cancelGeneration = () => {
    if (eventSource) {
      eventSource.close();
    }
  };

  const evaluatePrompt = (promptOptions: PromptOptions) => {
    console.log("OPENING EVENT SOURCE");
    const url = `${TREE_ENDPOINT}?topP=${promptOptions.topP}&maxBeams=${promptOptions.maxBeams}&maxNewTokens=${promptOptions.maxNewTokens}&prompt=${promptOptions.prompt}`;
    const eventSource = new EventSource(url);
    setLastPromptOptions(promptOptions);
    setLevels([]);

    eventSource.onerror = (event) => {
      console.error("EventSource error:", event);
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
    };

    setEventSource(eventSource);
  };

  return (
    <div className="App-root">
      <header className="App-header">
        <h1>Divergent Beams</h1>
      </header>
      <div className="App-content">
        <PromptInput
          lastPromptOptions={lastPromptOptions}
          evaluatePrompt={evaluatePrompt}
        ></PromptInput>
        <GenTree
          levels={levels}
          isGenerating={!!eventSource && eventSource.readyState === 1}
          cancelGeneration={cancelGeneration}
        ></GenTree>
        <Generations></Generations>
      </div>
    </div>
  );
}

export default App;
