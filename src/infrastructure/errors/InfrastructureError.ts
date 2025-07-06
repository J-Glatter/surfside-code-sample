export class InfrastructureError extends Error {
  constructor(
    public name: string,
    public entity: string,
    public message: string,
    public context: Record<string, unknown>,
  ) {
    super(message);
  }
}
