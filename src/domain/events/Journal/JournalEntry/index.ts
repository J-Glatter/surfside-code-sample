import { JournalEntryAdded } from './JournalEntryAdded';
import { JournalEntryAnalysed } from './JournalEntryAnalysedEvent';

export * from './JournalEntryAdded';
export * from './JournalEntryAnalysedEvent';

export type JournalEntryEvent = JournalEntryAdded | JournalEntryAnalysed;
