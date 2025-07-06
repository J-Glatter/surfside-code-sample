export class ApplicationError extends Error {
  constructor(
    public name: string,
    public message: string,
    public context: Record<string, unknown>,
  ) {
    super(message);
  }
}
