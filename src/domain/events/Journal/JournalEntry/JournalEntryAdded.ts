import { DomainEvent } from '../../Event';
import { JournalEntryProps } from '../../../entities';

export type JournalEntryAddedPayload = JournalEntryProps;

export class JournalEntryAdded extends DomainEvent<JournalEntryAddedPayload> {
  public static readonly eventName: string = 'JournalEntryAdded';

  constructor(payload: JournalEntryAddedPayload) {
    super(payload, JournalEntryAdded.name);
  }
}
