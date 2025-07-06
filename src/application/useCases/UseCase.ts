import { z, ZodSchema } from 'zod';
import { InvalidPayload } from '../errors';

export interface UseCase<Payload = {}, Response = {}> {
  execute(payload: Payload): Promise<Response>;
}

type ZodIssue = {
  path: string[];
  message: string;
  expected: string;
  received: string;
};

export abstract class ValidatedUseCase<P, R> implements UseCase<P, R> {
  protected abstract schema: ZodSchema<P>;

  protected abstract run(payload: P): Promise<R>;

  private formatIssue(issue: ZodIssue) {
    return `${issue.path.join('.')}: ${issue.message} ${issue.expected ? `, expected ${issue.expected} but received ${issue.received}` : ''}`.trim();
  }

  async execute(payload: P): Promise<R> {
    try {
      this.schema.parse(payload);
      return this.run(payload);
    } catch (error) {
      if (error instanceof z.ZodError) {
        throw new InvalidPayload({
          message: error.message,
          errors: error.errors.map((e) => this.formatIssue(e as ZodIssue)),
        });
      }
      throw error;
    }
  }
}
