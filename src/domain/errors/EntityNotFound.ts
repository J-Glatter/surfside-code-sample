import { DomainError } from './DomainError';

export class EntityNotFound extends DomainError {
  name = 'EntityNotFound';

  constructor({
    entityType,
    entityId,
  }: {
    entityType: string;
    entityId: string;
  }) {
    super(
      EntityNotFound.name,
      `${entityType} with id ${entityId} was not found`,
      entityId,
      {
        entityType,
        entityId,
      },
    );
  }
}
