import { JournalEntry } from '../entities';
import { EntityNotFound } from '../errors';
import { Aggregate } from './Aggregate';
import {
  JournalEntryAdded,
  JournalEvent,
  JournalEntryAnalysed,
} from '../events';
import { Analysis } from '../valueObjects';

export type JournalProps = {
  id: string;
  userId: string;
  createdAt: Date;
  lastUpdated: Date;
  entries: JournalEntry[];
};

export class Journal extends Aggregate<JournalEvent> {
  private readonly _id: JournalProps['id'];
  private readonly _userId: JournalProps['userId'];
  private _entries: JournalProps['entries'];
  private readonly _createdAt: Date;
  private readonly _lastUpdated: Date;

  private constructor(props: JournalProps) {
    super();
    this._id = props.id;
    this._userId = props.userId;
    this._entries = props.entries || [];
    this._createdAt = props.createdAt || [];
    this._lastUpdated = props.lastUpdated;
  }

  public static fromPartial(
    props: Pick<JournalProps, 'id' | 'userId' | 'createdAt' | 'lastUpdated'> & {
      entries: JournalEntry[];
    },
  ): Journal {
    return new Journal({
      id: props.id,
      userId: props.userId,
      createdAt: props.createdAt,
      lastUpdated: props.lastUpdated,
      entries: props.entries,
    });
  }

  public static create(props: Pick<JournalProps, 'id' | 'userId'>): Journal {
    const createdAt = new Date();

    const journal = new Journal({
      ...props,
      entries: [],
      lastUpdated: new Date(),
      createdAt,
    });

    // journal.addEvent(new JournalCreated({createdAt, userId: props.userId, id: props.id}))

    return journal;
  }

  public static fromExisting(props: JournalProps): Journal {
    return new Journal(props);
  }

  public analyseEntry(entryId: string, analysis: Analysis): void {
    const entry = this._entries.find(
      (entry: JournalEntry) => entry.entryId === entryId,
    );

    if (!entry)
      throw new EntityNotFound({
        entityId: entryId,
        entityType: 'JournalEntry',
      });

    entry.addAnalysis(analysis);
    this.addEvent(new JournalEntryAnalysed(entry.details));
  }

  public addEntry(entry: JournalEntry): void {
    this._entries = [...this._entries, entry];
    this.addEvent(new JournalEntryAdded(entry));
  }

  public getEntriesBetween({
    from,
    to,
  }: {
    from: Date;
    to: Date;
  }): JournalEntry[] {
    return this._entries.filter(
      (entry) => entry.createdAt >= from && entry.createdAt <= to,
    );
  }

  get entries(): JournalEntry[] {
    return this._entries;
  }

  get userId(): string {
    return this._userId;
  }

  get lastUpdated(): Date {
    return this._lastUpdated;
  }

  get id(): string {
    return this._id;
  }

  get createdAt() {
    return this._createdAt;
  }

  get details() {
    return {
      id: this._id,
      userId: this._userId,
      createdAt: this._createdAt,
      lastUpdated: this._lastUpdated,
      entries: this._entries,
    };
  }
}
