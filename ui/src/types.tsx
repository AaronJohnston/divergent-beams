export type LevelSpec = {
  id: string;
  level_type: "k_means" | "top_p";
  nodes: NodeSpec[];
};

export type NodeSpec = {
  content: string;
  prob: number;
  parent?: number;
  aunts?: number[];
  status?: string[];
};

export type PromptOptions = {
  prompt: string;
  topP: number;
  topK: number;
  maxBeams: number;
  maxNewTokens: number;
  topPDecay: number;
};
