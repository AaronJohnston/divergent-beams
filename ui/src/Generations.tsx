import Generation from "./Generation";
import { FinishedSpec, LevelSpec } from "./types";
import { logSumExp } from "./utils";

function Generations({
  levels,
  finished,
}: {
  levels: LevelSpec[];
  finished: FinishedSpec[];
}) {
  const generations: { content: string; prob: number }[] = [];

  if (levels.length > 0) {
    const minValue =
      -0.1 *
      levels[levels.length - 1].nodes.reduce(
        (acc, node) => Math.min(acc, node.prob),
        0
      ); // Smooth out normalized probabilities
    const logSumExpValue = logSumExp(
      levels[levels.length - 1].nodes.map((node) => node.prob / minValue)
    );

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
        prob: Math.exp(lastNode.prob / minValue - logSumExpValue),
      });
    }

    generations.sort((a, b) => b.prob - a.prob);
  }

  return (
    <div className="Generations horiz-scroll">
      {finished.map((finished) => {
        return (
          <Generation
            key={finished.content}
            content={finished.content}
            prob={finished.prob}
            finished={true}
          />
        );
      })}
      {generations.map((generation) => {
        return (
          <Generation
            key={generation.content}
            content={generation.content}
            prob={generation.prob}
            finished={false}
          />
        );
      })}
    </div>
  );
}

export default Generations;
