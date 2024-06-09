import { useState } from "react";
import "./App.css";
import GenTree from "./GenTree";
import PromptInput from "./PromptInput";
import { LevelSpec } from "./types";

const TREE_ENDPOINT =
  "http://ec2-52-89-34-232.us-west-2.compute.amazonaws.com/api/v1/tree";

function App() {
  const [levels, setLevels] = useState<LevelSpec[]>([]);

  const evaluatePrompt = () => {
    console.log("OPENING EVENT SOURCE");
    const eventSource = new EventSource(TREE_ENDPOINT + "?prompt=" + prompt);
    setLevels([]);

    eventSource.onerror = (event) => {
      console.error("EventSource failed:", event);
      eventSource.close();
    };

    eventSource.addEventListener("level", (event) => {
      if (event.lastEventId === "END") {
        console.log("CLOSING EVENT SOURCE");
        eventSource.close();
        return;
      }
      console.log(event);
      const data: LevelSpec = JSON.parse(event.data);
      setLevels((levels: LevelSpec[]) => [...levels, data]);
    });

    return () => {
      console.log("CLEANING UP EVENT SOURCE");
      eventSource.close();
    };
  };

  return (
    <div className="App-root">
      <header className="App-header">
        <h1>LLM Output Space</h1>
      </header>
      <PromptInput evaluatePrompt={evaluatePrompt}></PromptInput>
      <GenTree levels={levels}></GenTree>
    </div>
  );
}

export default App;
