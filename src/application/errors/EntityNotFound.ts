import { ApplicationError } from './ApplicationError';

type ErrorProps = {
  entity: { name: string; id: string };
  message?: string;
  context?: Record<string, unknown>;
};

export class EntityNotFound extends ApplicationError {
  name = 'EntityNotFound';

  constructor({ entity, message, context }: ErrorProps) {
    super(
      EntityNotFound.name,
      `${entity.name}: ${entity.id} not found${message ? `; ${message}` : ''}`,
      context ? context : entity,
    );
  }
}
