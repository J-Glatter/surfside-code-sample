import {
  FindEntryValues,
  JournalEntry,
  JournalEntryRepository,
} from '../../../domain';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';
import { inject, injectable } from 'tsyringe';
import { MultipleItemsFound } from '../../errors';

@injectable()
export class DynamoJournalEntryRepository implements JournalEntryRepository {
  private readonly docClient: DynamoDBDocumentClient;
  private readonly tableName: string;

  constructor(
    @inject('DynamoDBDocumentClient') docClient: DynamoDBDocumentClient,
    @inject('journalTableName') tableName: string,
  ) {
    this.docClient = docClient;
    this.tableName = tableName;
  }

  public async findEntry({
    entryId,
    userId,
    journalId,
    createdAt,
  }: FindEntryValues): Promise<JournalEntry> {
    const result = await this.docClient.send(
      new QueryCommand({
        TableName: this.tableName,
        KeyConditionExpression: 'PK = :pk and SK = :sk',
        ExpressionAttributeValues: {
          ':pk': `USER#${userId}`,
          ':sk': `JOURNAL#${journalId}#ENTRY#${createdAt}#${entryId}`,
        },
      }),
    );

    const items = result.Items || [];

    if (items.length > 1) {
      throw new MultipleItemsFound({
        entryId,
        userId,
        journalId,
        createdAt,
      });
    }

    return JournalEntry.fromExisting({
      entryId,
      journalId,
      userId,
      createdAt: new Date(createdAt),
      entryContent: items[0].entryContent,
      summary: items[0].summary,
      tags: items[0].tags,
      analysedAt: new Date(items[0].analysedAt),
    });
  }
}
