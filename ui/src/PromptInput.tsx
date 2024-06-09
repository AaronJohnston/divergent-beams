import { useState } from "react";

function PromptInput({
  evaluatePrompt,
}: {
  evaluatePrompt: (prompt: string) => void;
}) {
  const [currPrompt, setCurrPrompt] = useState(
    "What is the closest star to the Earth?"
  );

  return (
    <div className="PromptInput-root">
      <textarea
        className="GenPromptTokens"
        value={currPrompt}
        onChange={(event) => setCurrPrompt(event.target.value)}
        onBlur={() => evaluatePrompt(currPrompt)}
      ></textarea>
    </div>
  );
}

export default PromptInput;
