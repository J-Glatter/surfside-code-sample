import { JournalEntry } from '../entities';

export type FindEntryValues = {
  entryId: string;
  userId: string;
  journalId: string;
  createdAt: string;
};

export interface JournalEntryRepository {
  findEntry(entryValues: FindEntryValues): Promise<JournalEntry>;
}
