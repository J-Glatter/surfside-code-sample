export class DomainError extends Error {
  constructor(
    public name: string,
    public message: string,
    public entityId: string,
    public context: Record<string, unknown>,
  ) {
    super(message);
  }
}
