import Generation from "./Generation";
import { LevelSpec } from "./types";
import { logSumExp } from "./utils";

function Generations({ levels }: { levels: LevelSpec[] }) {
  const generations: { content: string; prob: number }[] = [];

  if (levels.length > 0) {
    const logSumExpValue = logSumExp(
      levels[levels.length - 1].nodes.map((node) => node.prob)
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

      console.log(
        lastNode.prob,
        logSumExpValue,
        Math.exp(lastNode.prob - logSumExpValue)
      );
      generations.push({
        content: generation
          .reverse()
          .join("")
          .replace(/▁/g, " ")
          .replace(/<0x0A>/g, "<br />"),
        prob: Math.exp(lastNode.prob - logSumExpValue),
      });
    }

    generations.sort((a, b) => b.prob - a.prob);
  }

  return (
    <div className="Generations horiz-scroll">
      {generations.map((generation) => {
        return (
          <Generation
            key={generation.content}
            content={generation.content}
            prob={generation.prob}
          />
        );
      })}
    </div>
  );
}

export default Generations;
