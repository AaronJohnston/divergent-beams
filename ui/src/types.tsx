export type LevelSpec = {
  id: string;
  level_type: "gather" | "sample";
  duration: number;
  nodes: NodeSpec[];
  finished: FinishedSpec[];
};

export type NodeSpec = {
  content: string;
  prob: number;
  parent?: number;
  aunts?: number[];
  status?: string[];
};

export type FinishedSpec = {
  content: string;
  prob: number;
  parent: number;
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
