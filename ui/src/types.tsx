export type LevelSpec = NodeSpec[];

export type NodeSpec = {
  content: string;
  prob: number;
  parent?: number;
  status?: string[];
};
