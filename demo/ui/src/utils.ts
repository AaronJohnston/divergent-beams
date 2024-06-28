export function logSumExp(values: number[]): number {
  const max = Math.max(...values);
  return (
    max +
    Math.log(values.reduce((acc, value) => acc + Math.exp(value - max), 0))
  );
}
