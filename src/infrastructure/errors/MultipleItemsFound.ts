import { InfrastructureError } from './InfrastructureError';

export class MultipleItemsFound extends InfrastructureError {
  name = 'MultipleItemsFound';

  constructor(searchParams: Record<string, unknown>) {
    super(
      MultipleItemsFound.name,
      'JournalEntry',
      `Multiple entries found when searching by ${JSON.stringify(searchParams)}`,
      searchParams,
    );
  }
}
