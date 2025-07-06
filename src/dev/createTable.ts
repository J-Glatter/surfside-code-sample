import {
  DynamoDBClient,
  CreateTableCommand,
  DescribeTableCommand,
} from '@aws-sdk/client-dynamodb';
import { TransactWriteCommand } from '@aws-sdk/lib-dynamodb';

const region = process.env.AWS_REGION || 'eu-west-1';
const endpoint = process.env.DYNAMODB_ENDPOINT || 'http://localhost:8000';
const tableName = process.env.TABLE_NAME || 'JournalTable';

const client = new DynamoDBClient({
  region,
  endpoint,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || 'testKey',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || 'testSecretKey',
  },
});

const testUser = '6baa4f27-c2b9-48ce-aeec-adad87835e7e';
const testJournal = '45009cdb-2827-4f8c-9ba7-670cfb05af01';

const createTestJournalCommand = new TransactWriteCommand({
  TransactItems: [
    {
      Put: {
        TableName: tableName,
        Item: {
          PK: `USER#${testUser}`,
          SK: `JOURNAL#${testJournal}`,
          createdAt: new Date().toISOString(),
          lastUpdated: new Date().toISOString(),
        },
      },
    },
  ],
});

async function createTable() {
  try {
    await client.send(new DescribeTableCommand({ TableName: tableName }));
    console.log(`Table ${tableName} already exists.`);
    await client.send(createTestJournalCommand);
  } catch (error: any) {
    if (error.name === 'ResourceNotFoundException') {
      console.log(`Table ${tableName} not found. Creating table...`);
      const command = new CreateTableCommand({
        TableName: tableName,
        AttributeDefinitions: [
          { AttributeName: 'PK', AttributeType: 'S' },
          { AttributeName: 'SK', AttributeType: 'S' },
        ],
        KeySchema: [
          { AttributeName: 'PK', KeyType: 'HASH' },
          { AttributeName: 'SK', KeyType: 'RANGE' },
        ],
        BillingMode: 'PAY_PER_REQUEST',
      });
      await client.send(command);
      console.log(`Table ${tableName} created successfully.`);

      await client.send(createTestJournalCommand);
    } else {
      console.error('Error describing table:', error);
      throw error;
    }
  }
}

createTable().catch((err) => {
  console.error('Failed to create table:', err);
  process.exit(1);
});
