export type LevelSpec = {
  id: string;
  level_type: "k_means" | "top_p";
  nodes: NodeSpec[];
};

export type NodeSpec = {
  content: string;
  prob: number;
  parents?: number[];
  status?: string[];
};

export type PromptOptions = {
  prompt: string;
  maxBeams: number;
  topP: number;
};
