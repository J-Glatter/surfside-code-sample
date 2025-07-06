import {
  DynamoDBDocumentClient,
  QueryCommand,
  TransactWriteCommand,
  TransactWriteCommandInput,
} from '@aws-sdk/lib-dynamodb';
import {
  JournalEntry,
  Journal,
  JournalEvent,
  JournalEntryAdded,
  JournalRepository,
  JournalEntryAnalysed,
} from '../../../domain';
import { inject, injectable } from 'tsyringe';
import { result, snakeCase } from 'lodash';

type EventToItemMapper = (
  event: JournalEvent,
  journal: Journal,
) => Record<string, unknown>;
type JournalItem = {
  PK: string;
  SK: string;
  lastUpdated: string;
  createdAt: string;
  entries: JournalEntry[];
};

type JournalEntryItem = {
  PK: string;
  SK: string;
  journalId: string;
  entryId: string;
  userId: string;
  entryContent: string;
  createdAt: string;
  summary: string;
  analysedAt: string | undefined;
  tags: string[];
};

const eventToItemMap: Record<string, EventToItemMapper> = {
  [JournalEntryAdded.name]: (event: JournalEvent) => {
    const entry = (event as JournalEntryAdded).payload;
    return {
      SK: `JOURNAL#${entry.journalId}#ENTRY#${entry.createdAt.toISOString()}#${entry.entryId}`,
      entryContent: entry.entryContent,
      createdAt: entry.createdAt.toISOString(),
      summary: entry.summary,
    };
  },
  [JournalEntryAnalysed.name]: (event: JournalEntryAnalysed) => {
    const { createdAt, ...entry } = (event as JournalEntryAnalysed).payload;
    return {
      SK: `JOURNAL#${entry.journalId}#ENTRY#${createdAt.toISOString()}#${entry.entryId}`,
      createdAt: createdAt.toISOString(),
      ...entry,
    };
  },
};

@injectable()
export class DynamoJournalRepository implements JournalRepository {
  private readonly docClient: DynamoDBDocumentClient;
  private readonly tableName: string;

  constructor(
    @inject('DynamoDBDocumentClient') docClient: DynamoDBDocumentClient,
    @inject('journalTableName') tableName: string,
  ) {
    this.docClient = docClient;
    this.tableName = tableName;
  }

  async save(journal: Journal): Promise<void> {
    const events = journal.uncommittedEvents;
    const transactItems: TransactWriteCommandInput['TransactItems'] = [];

    events.forEach((event: JournalEvent) => {
      const mapper = eventToItemMap[event.constructor.name];
      if (mapper) {
        const item = mapper(event, journal);
        transactItems.push({
          Put: {
            TableName: this.tableName,
            Item: { PK: `USER#${journal.userId}`, ...item },
          },
        });
        transactItems.push({
          Put: {
            TableName: this.tableName,
            Item: {
              PK: `EVENT#${event.id}`,
              SK: snakeCase(event.eventName),
              payload: event.payload,
              occurredAt: event.occurredAt,
            },
          },
        });
      } else {
        throw new Error(`Unhandled event type: ${event.constructor.name}`);
      }
    });

    if (transactItems.length > 0) {
      await this.docClient.send(
        new TransactWriteCommand({
          TransactItems: transactItems,
        }),
      );
      journal.clearEvents();
    }
  }

  async findById(
    {
      journalId,
      userId,
    }: {
      journalId: string;
      userId: string;
    },
    includeEntries: boolean,
  ): Promise<Journal | null> {
    const result = await this.docClient.send(
      new QueryCommand({
        TableName: this.tableName,
        KeyConditionExpression: 'PK = :pk and SK = :sk',
        ExpressionAttributeValues: {
          ':pk': `USER#${userId}`,
          ':sk': `JOURNAL#${journalId}`,
        },
      }),
    );

    if (!result.Items || result.Items.length === 0) {
      return null;
    }

    if (result.Items && result.Items.length > 0) {
      const journal = result.Items[0];

      if (!includeEntries)
        return this.journalItemToJournal(journal as JournalItem);

      const rawData = {
        ...journal,
        entries: await this.getJournalEntries(journalId, userId),
      };

      return this.journalItemToJournal(rawData as JournalItem);
    }

    return null;
  }

  private async getJournalEntries(
    journalId: string,
    userId: string,
  ): Promise<JournalEntry[]> {
    const { Items } = await this.docClient.send(
      new QueryCommand({
        TableName: this.tableName,
        KeyConditionExpression: 'PK = :pk and begins_with(SK, :entryPrefix)',
        ExpressionAttributeValues: {
          ':pk': `USER#${userId}`,
          ':entryPrefix': `JOURNAL#${journalId}#ENTRY`,
        },
      }),
    );

    return ((Items as JournalEntryItem[]) || []).map((item) =>
      JournalEntry.fromExisting({
        ...item,
        userId,
        createdAt: new Date(item.createdAt),
        analysedAt: item.analysedAt ? new Date(item.analysedAt) : undefined,
      }),
    );
  }

  private journalItemToJournal = (data: JournalItem): Journal => {
    const userId = data.PK.split('#')[1];
    const journalId = data.SK.split('#')[1];
    const lastUpdated = new Date(data.lastUpdated);
    const createdAt = new Date(data.createdAt);

    const entries = data.entries || [];

    return Journal.fromExisting({
      id: journalId,
      userId,
      lastUpdated,
      entries,
      createdAt,
    });
  };
}
