import Generation from "./Generation";
import { FinishedSpec, LevelSpec } from "./types";
import { logSumExp } from "./utils";

function Generations({
  levels,
  finished,
  isGenerationComplete,
}: {
  levels: LevelSpec[];
  finished: FinishedSpec[];
  isGenerationComplete: boolean;
}) {
  const generations: { content: string; prob: number; isActive: boolean }[] =
    [];

  for (const finishedGeneration of finished) {
    generations.push({
      content: finishedGeneration.content,
      prob: finishedGeneration.prob,
      isActive: false,
    });
  }

  if (levels.length > 0) {
    for (const lastNode of levels[levels.length - 1].nodes) {
      let current = lastNode;
      const generation = [current.content];

      for (let i = levels.length - 2; i >= 0; i -= 1) {
        if (current.parent === undefined) {
          break;
        }

        const next = levels[i].nodes[current.parent];

        if (levels[i].level_type === "sample") {
          generation.push(next.content);
        }

        current = next;
      }

      generations.push({
        content: generation
          .reverse()
          .join("")
          .replace(/▁/g, " ")
          .replace(/<0x0A>/g, "\n"),
        prob: lastNode.prob,
        isActive: true,
      });
    }
  }

  generations.sort((a, b) => b.prob - a.prob);

  const minValue =
    -0.1 *
    generations.reduce((acc, generation) => Math.min(acc, generation.prob), 0); // Smooth out normalized probabilities
  const logSumExpValue = logSumExp(
    generations.map((node) => node.prob / minValue)
  );

  const normalizedGenerations = generations.map((generation) => {
    return {
      ...generation,
      prob: Math.exp(generation.prob / minValue - logSumExpValue),
    };
  });

  return (
    <div className="Generations horiz-scroll">
      {normalizedGenerations.map((generation) => {
        return (
          <Generation
            key={generation.content}
            content={generation.content}
            prob={generation.prob}
            isActive={generation.isActive}
            isGenerationComplete={isGenerationComplete}
          />
        );
      })}
    </div>
  );
}

export default Generations;
