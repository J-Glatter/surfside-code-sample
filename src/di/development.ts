import { container } from 'tsyringe';
import {
  JournalRepository,
  JournalAnalysisService,
  JournalEntryRepository,
} from '../domain';
import {
  DynamoJournalEntryRepository,
  DynamoJournalReadRepository,
  DynamoJournalRepository,
  StubbedJournalAnalysisService,
} from '../infrastructure';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';
import { JournalReadRepository } from '../application/queries';

const appConfig = {
  tableName: process.env.TABLE_NAME || 'JournalTable',
  region: process.env.AWS_REGION || 'eu-west-1',
  dynamoDBEndpoint: process.env.DYNAMO_ENDPOINT,
};

const dynamoClient = new DynamoDBClient({
  region: process.env.AWS_REGION || 'eu-west-1',
  endpoint: process.env.DYNAMODB_ENDPOINT || 'http://localhost:8000',
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || 'testKey',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || 'testSecretKey',
  },
});

const docClient = DynamoDBDocumentClient.from(dynamoClient, {
  marshallOptions: {
    convertClassInstanceToMap: true,
  },
});

container.registerInstance<DynamoDBDocumentClient>(
  'DynamoDBDocumentClient',
  docClient,
);
container.register<string>('journalTableName', {
  useValue: appConfig.tableName,
});

container.register<JournalRepository>('JournalRepository', {
  useClass: DynamoJournalRepository,
});

container.register<JournalReadRepository>('JournalReadRepository', {
  useClass: DynamoJournalReadRepository,
});

container.register<JournalEntryRepository>('JournalEntryRepository', {
  useClass: DynamoJournalEntryRepository,
});

container.register<JournalAnalysisService>('JournalAnalysisService', {
  useClass: StubbedJournalAnalysisService,
});

container.register('AppConfig', { useValue: appConfig });
