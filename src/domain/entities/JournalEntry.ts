type Analysis = {
  summary: string;
};

export type JournalEntryProps = {
  journalId: string;
  entryId: string;
  userId: string;
  entryContent: string;
  createdAt: Date;
  tags: string[];
  analysedAt?: Date;
  summary: string;
};

type JournalEntryCreatedProps = Omit<
  JournalEntryProps,
  'summary' | 'analysedAt'
>;

export class JournalEntry implements JournalEntryProps {
  public readonly entryId: string;
  public readonly journalId: string;
  public readonly userId: string;
  public readonly entryContent: string;
  public readonly createdAt: Date;
  public summary: string;
  public readonly tags: string[];
  public analysedAt: Date | undefined;

  private constructor({
    entryId,
    journalId,
    userId,
    entryContent,
    createdAt,
    summary,
    tags,
  }: JournalEntryProps) {
    this.entryId = entryId;
    this.journalId = journalId;
    this.userId = userId;
    this.entryContent = entryContent;
    this.createdAt = createdAt;
    this.summary = summary;
    this.tags = tags;
  }

  public static fromExisting(journalEntry: JournalEntryProps): JournalEntry {
    return new JournalEntry(journalEntry);
  }

  public static create(
    props: Omit<
      JournalEntryCreatedProps,
      'createdAt' | 'summary' | 'analysedAt'
    >,
  ): JournalEntry {
    return new JournalEntry({
      ...props,
      createdAt: new Date(),
      summary: '',
      analysedAt: undefined,
    });
  }

  public addAnalysis(analysis: Analysis) {
    this.summary = analysis.summary;
    this.analysedAt = new Date();
  }

  public get details(): JournalEntryProps {
    return {
      journalId: this.journalId,
      entryId: this.entryId,
      userId: this.userId,
      entryContent: this.entryContent,
      createdAt: this.createdAt,
      summary: this.summary,
      tags: this.tags,
      analysedAt: this.analysedAt,
    };
  }
}
