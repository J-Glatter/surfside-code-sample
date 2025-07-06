import { JournalEntry } from '../../domain';

export interface JournalReadRepository {
  findEntries(
    journalId: string,
    userId: string,
    opts?: { between?: { fromDate: Date; toDate: Date } },
  ): Promise<JournalEntry[]>;
}
