import { Journal } from '../aggregates';

export interface JournalRepository {
  save(journal: Journal): Promise<void>;
  findById(
    ids: { journalId: string; userId: string },
    includeEntries?: boolean,
  ): Promise<Journal | null>;
}
