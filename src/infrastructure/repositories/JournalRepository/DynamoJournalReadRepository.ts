import { JournalReadRepository } from '../../../application/queries';
import { JournalEntry } from '../../../domain';
import {
  DynamoDBDocumentClient,
  QueryCommand,
  QueryCommandInput,
} from '@aws-sdk/lib-dynamodb';
import { inject, injectable } from 'tsyringe';

@injectable()
export class DynamoJournalReadRepository implements JournalReadRepository {
  private readonly docClient: DynamoDBDocumentClient;
  private readonly tableName: string;

  constructor(
    @inject('DynamoDBDocumentClient') docClient: DynamoDBDocumentClient,
    @inject('journalTableName') tableName: string,
  ) {
    this.docClient = docClient;
    this.tableName = tableName;
  }

  public async findEntries(
    journalId: string,
    userId: string,
    opts?: { between?: { fromDate: Date; toDate: Date } },
  ): Promise<JournalEntry[]> {
    const pk = `USER#${userId}`;
    const params: QueryCommandInput = {
      TableName: this.tableName,
      KeyConditionExpression: 'PK = :pk',
      ExpressionAttributeValues: { ':pk': pk },
    };
    if (opts?.between) {
      const fromISO = opts.between.fromDate.toISOString();
      const toISO = opts.between.toDate.toISOString();

      params.KeyConditionExpression += ' AND begins_with(SK, :entryPrefix)';
      params.ExpressionAttributeValues![':entryPrefix'] =
        `JOURNAL#${journalId}#ENTRY#`;

      params.FilterExpression = 'createdAt BETWEEN :fromDate AND :toDate';
      params.ExpressionAttributeValues![':fromDate'] = fromISO;
      params.ExpressionAttributeValues![':toDate'] = toISO;
    } else {
      params.KeyConditionExpression += ' AND begins_with(SK, :entryPrefix)';
      params.ExpressionAttributeValues![':entryPrefix'] =
        `JOURNAL#${journalId}#ENTRY#`;
    }

    const result = await this.docClient.send(new QueryCommand(params));
    const items = result.Items || [];

    return items.map((item) =>
      JournalEntry.fromExisting({
        entryId: item.SK.split('#')[3],
        journalId,
        userId,
        entryContent: item.entryContent,
        createdAt: new Date(item.createdAt),
        summary: item.summary,
        tags: item.tags,
        analysedAt: new Date(item.analysedAt),
      }),
    );
  }
}
