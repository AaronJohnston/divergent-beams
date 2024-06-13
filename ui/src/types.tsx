export type LevelSpec = {
  id: string;
  level_type: "gather" | "sample";
  duration: number;
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
  gatherAlgo: string;
};
