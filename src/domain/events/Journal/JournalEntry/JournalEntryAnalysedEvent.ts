import { DomainEvent } from '../../Event';
import { Analysis } from '../../../valueObjects';
import { JournalEntry, JournalEntryProps } from '../../../entities';

export type JournalEntryAnalysedPayload = JournalEntryProps;

export class JournalEntryAnalysed extends DomainEvent<JournalEntryAnalysedPayload> {
  static readonly eventName = 'JournalEntryAnalysed';
  constructor(payload: JournalEntryAnalysedPayload) {
    super(payload, JournalEntryAnalysed.name);
  }
}
