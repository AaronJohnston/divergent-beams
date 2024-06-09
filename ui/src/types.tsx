export type LevelSpec = NodeSpec[];

export type NodeSpec = {
  content: string;
  prob: number;
  parent?: number;
  status?: string[];
};

export type PromptOptions = {
  prompt: string;
  maxBeams: number;
  topP: number;
};
